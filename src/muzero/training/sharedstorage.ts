import { MuZeroNet } from '../networks/fullconnected'

import debugFactory from 'debug'
import { BaseMuZeroNet } from '../networks/network'
const debug = debugFactory('muzero:sharedstorage:module')

export class MuZeroSharedStorage {
  private readonly networks: Map<number, BaseMuZeroNet>

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
    this.networks = new Map<number, BaseMuZeroNet>()
  }

  public async latestNetwork (): Promise<BaseMuZeroNet> {
    if (this.networks.size > 0) {
      const keys: number[] = []
      this.networks.forEach((v, k) => keys.push(k))
      keys.sort((a, b) => b - a)
      debug('Picked the network with id %d', keys[0])
      const network = this.networks.get(keys[0])
      if (network !== undefined) {
        return network
      } else {
        throw new Error(`Unable to get latest network '${keys[0]}'`)
      }
    } else {
      // make uniform network: policy -> uniform, value -> 0, reward -> 0
      const network = new MuZeroNet(this.config.observationSize, this.config.actionSpaceSize)
      try {
        debug('Loading network')
        await network.load('file://data/muzeronet')
      } catch (e) {
        debug(e)
      }
      return network
    }
  }

  public async saveNetwork (step: number, network: BaseMuZeroNet): Promise<void> {
    debug('Saving network')
    await network.save('file://data/muzeronet')
    this.networks.set(step, network)
  }
}
