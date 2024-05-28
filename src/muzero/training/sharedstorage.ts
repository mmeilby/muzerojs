import debugFactory from 'debug'
import { type Config } from '../games/core/config'
import { type Network } from '../networks/nnet'
import { UniformNetwork } from '../networks/implementations/uniform'

import { EventEmitter } from 'events'
import { CoreNet } from '../networks/implementations/core'
import { ResNet } from '../networks/implementations/conv'
import fs from 'fs'

const debug = debugFactory('muzero:sharedstorage:module')

export class SharedStorage {
  public networkCount: number
  private readonly path: string
  private readonly latestNetwork_: Network
  //  private readonly maxNetworks: number
  private readonly updatedNetworkEvent: EventEmitter

  /**
   *
   * @param config Network configuration needed for creation of Neural Networks
   * @param network
   * @param config.observationSize Length of observation tensors (input for the representation model h)
   * @param config.actionSpaceSize Length of the action tensors (partial input for the dynamics model g)
   */
  constructor (
    private readonly config: Config,
    network?: Network
  ) {
    this.latestNetwork_ = network ?? this.initialize()
    //    this.maxNetworks = 2
    this.updatedNetworkEvent = new EventEmitter()
    this.networkCount = (network != null) ? 0 : -1
    this.path = 'data/'.concat(this.config.savedNetworkPath, '/')
    // Create path if needed
    fs.stat(this.path, (err, _stats) => {
      if (err != null) {
        fs.mkdir(this.path, () => {
          debug(`Created network data path: ${this.path}`)
        })
      }
    })
  }

  public initialize (): Network {
    const model = new ResNet(
      this.config.observationSize,
      this.config.actionSpace,
      this.config.observationSize,
      this.config.actionShape
    )
    return new CoreNet(model, this.config)
  }

  public uniformNetwork (): Network {
    // make uniform network: policy -> uniform, value -> 0, reward -> 0
    return new UniformNetwork(this.config.actionSpace)
  }

  public latestNetwork (): Network {
    debug(`Picked the latest network - training step ${this.networkCount}`)
    return this.networkCount >= 0 ? this.latestNetwork_ : this.uniformNetwork()
  }

  public async waitForUpdate (): Promise<Network> {
    const promise = new Promise<Network>((resolve, _reject) => {
      this.updatedNetworkEvent.once('update_event', () => {
        resolve(this.latestNetwork())
      })
    })
    return await promise
  }

  public async loadNetwork (): Promise<void> {
    try {
      debug('Loading network')
      await this.latestNetwork_.load(`file://${this.path}`)
      this.networkCount = 0
    } catch (e) {
      debug(e)
    }
  }

  public async saveNetwork (step: number, network: Network): Promise<void> {
    debug('Saving network')
    await network.save(`file://${this.path}`)
    network.copyWeights(this.latestNetwork_)
    this.networkCount = step
    this.updatedNetworkEvent.emit('update_event')
  }
}
