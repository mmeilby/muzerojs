import { MuZeroNet } from '../networks/fullconnected'
import { BaseMuZeroNet } from '../networks/network'
import debugFactory from 'debug'

const debug = debugFactory('muzero:sharedstorage:module')

export class MuZeroSharedStorage {
  private readonly latestNetwork_: BaseMuZeroNet
  private readonly maxNetworks: number
  public networkCount: number

  /**
   *
   * @param config Network configuration needed for creation of Neural Networks
   * @param config.observationSize Length of observation tensors (input for the representation model h)
   * @param config.actionSpaceSize Length of the action tensors (partial input for the dynamics model g)
   */
  constructor (
    private readonly config: {
      observationSize: number
      actionSpaceSize: number
    }
  ) {
    this.latestNetwork_ = this.uniformNetwork()
    this.maxNetworks = 2
    this.networkCount = 0
  }

  public uniformNetwork (learningRate?: number): BaseMuZeroNet {
    // make uniform network: policy -> uniform, value -> 0, reward -> 0
    return new MuZeroNet(this.config.observationSize, this.config.actionSpaceSize, learningRate ?? 0)
  }

  public latestNetwork (): BaseMuZeroNet {
    debug(`Picked the latest network - training step ${this.networkCount}`)
    return this.latestNetwork_
  }

  public async loadNetwork (): Promise<void> {
    try {
      debug('Loading network')
      await this.latestNetwork_.load('file://data/')
    } catch (e) {
      debug(e)
    }
  }

  public async saveNetwork (step: number, network: BaseMuZeroNet): Promise<void> {
    debug('Saving network')
    await network.save('file://data/')
    network.copyWeights(this.latestNetwork_)
    this.networkCount = step
  }
}
