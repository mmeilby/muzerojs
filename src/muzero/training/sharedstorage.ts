import debugFactory from 'debug'
import { Config } from '../games/core/config'
import { Network } from '../networks/nnet'
import { MuZeroNet } from '../networks/network'
import { UniformNetwork } from '../networks/uniform'

const debug = debugFactory('muzero:sharedstorage:module')

export class SharedStorage {
  private readonly latestNetwork_: Network
  private readonly maxNetworks: number
  public networkCount: number

  /**
   *
   * @param config Network configuration needed for creation of Neural Networks
   * @param config.observationSize Length of observation tensors (input for the representation model h)
   * @param config.actionSpaceSize Length of the action tensors (partial input for the dynamics model g)
   */
  constructor (
    private readonly config: Config
  ) {
    this.latestNetwork_ = this.initialize()
    this.maxNetworks = 2
    this.networkCount = -1
  }

  public initialize (): Network {
    return new MuZeroNet(this.config.observationSize, this.config.actionSpace, this.config.lrInit)
  }

  public uniformNetwork (): Network {
    // make uniform network: policy -> uniform, value -> 0, reward -> 0
    return new UniformNetwork(this.config.actionSpace)
  }

  public latestNetwork (): Network {
    debug(`Picked the latest network - training step ${this.networkCount}`)
    return this.networkCount >= 0 ? this.latestNetwork_ : this.uniformNetwork()
  }

  public async loadNetwork (): Promise<void> {
    try {
      debug('Loading network')
      await this.latestNetwork_.load('file://data/'.concat(this.config.savedNetworkPath, '/'))
      this.networkCount = 0
    } catch (e) {
      debug(e)
    }
  }

  public async saveNetwork (step: number, network: Network): Promise<void> {
    debug('Saving network')
    await network.save('file://data/'.concat(this.config.savedNetworkPath, '/'))
    network.copyWeights(this.latestNetwork_)
    this.networkCount = step
  }
}
