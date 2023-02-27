import debugFactory from 'debug'
import { MuZeroConfig } from '../games/core/config'
import { MuZeroNetwork } from '../networks/nnet'
import { Actionwise } from '../selfplay/entities'
import { MuZeroNet } from '../networks/network'
import { MuZeroUniformNetwork } from '../networks/uniform'
import {Observation} from "../../alphazero/networks/nnet";
import {NetworkOutput} from "../../alphazero/networks/networkoutput";
import {MuZeroNetObservation} from "../../alphazero/networks/network";

const debug = debugFactory('muzero:sharedstorage:module')

export class MuZeroSharedStorage<Action extends Actionwise> {
  private readonly latestNetwork_: MuZeroNetwork<Action>
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
    this.latestNetwork_ = this.initialize()
    this.maxNetworks = 2
    this.networkCount = -1
  }

  public initialize (): MuZeroNetwork<Action> {
    return new MuZeroNet<Action>(this.config.observationSize, this.config.actionSpace, this.config.lrInit)
  }

  public uniformNetwork (): MuZeroNetwork<Action> {
    // make uniform network: policy -> uniform, value -> 0, reward -> 0
    return new MuZeroUniformNetwork(this.config.actionSpace)
  }

  public latestNetwork (): MuZeroNetwork<Action> {
    debug(`Picked the latest network - training step ${this.networkCount}`)
    return this.networkCount >= 0 ? this.latestNetwork_ : this.uniformNetwork()
  }

  public async loadNetwork (): Promise<void> {
    try {
      debug('Loading network')
      await this.latestNetwork_.load('file://data/')
      this.networkCount = 0
    } catch (e) {
      debug(e)
    }
  }

  public async saveNetwork (step: number, network: MuZeroNetwork<Action>): Promise<void> {
    debug('Saving network')
    await network.save('file://data/')
    network.copyWeights(this.latestNetwork_)
    this.networkCount = step
  }
}
