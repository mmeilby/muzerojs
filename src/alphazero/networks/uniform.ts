import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { Network, Observation } from './nnet'
import { Actionwise } from '../games/core/actionwise'
import { Statewise } from '../games/core/statewise'

export class UniformNetwork<State extends Statewise, Action extends Actionwise> implements Network<Action> {
  constructor (
    // Length of the action tensors
    private readonly actionSpace: number
  ) {}

  public initialInference (obs: Observation): NetworkOutput {
    const policy = new Array<number>(this.actionSpace).fill(1 / this.actionSpace)
    return new NetworkOutput(0, policy)
  }

  public async trainInference (samples: Array<Batch<Actionwise>>): Promise<number> {
    // A uniform network should never be trained
    throw new Error('Training has been attempted on a uniform mocked network. This is not allowed.')
  }

  public async save (path: string): Promise<void> {
    // No reason for saving anything from a uniform network
  }

  public async load (path: string): Promise<void> {
    // We can't load any data to a uniform network
    throw new Error('Load weights has been attempted on a uniform mocked network. This is not allowed.')
  }

  public copyWeights (network: Network<Action>): void {
    // A uniform network does not have any data to copy - leave the target network untouched
  }
}
