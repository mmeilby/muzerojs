import debugFactory from 'debug'
import {MuZeroConfig} from "../games/core/config";
import {MuZeroNetwork} from "../networks/nnet";
import {Actionwise} from "../selfplay/entities";
import {MuZeroNet} from "../networks/network";

const debug = debugFactory('muzero:sharedstorage:module')

export class MuZeroSharedStorage {
  private readonly latestNetwork_: MuZeroNetwork<Actionwise>
  private readonly maxNetworks: number
  public networkCount: number

  /**
   *
   * @param config Network configuration needed for creation of Neural Networks
   * @param config.observationSize Length of observation tensors (input for the representation model h)
   * @param config.actionSpaceSize Length of the action tensors (partial input for the dynamics model g)
   */
  constructor (
    private readonly config: MuZeroConfig
  ) {
    this.latestNetwork_ = this.uniformNetwork()
    this.maxNetworks = 2
    this.networkCount = 0
  }

  public uniformNetwork (learningRate?: number): MuZeroNetwork<Actionwise> {
    // make uniform network: policy -> uniform, value -> 0, reward -> 0
    return new MuZeroNet(this.config.observationSize, this.config.actionSpace, learningRate ?? 0)
    //TODO: Change this to a uniform mocked network
  }

  public latestNetwork (): MuZeroNetwork<Actionwise> {
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

  public async saveNetwork (step: number, network: MuZeroNetwork<Actionwise>): Promise<void> {
    debug('Saving network')
    await network.save('file://data/')
    network.copyWeights(this.latestNetwork_)
    this.networkCount = step
  }
}
