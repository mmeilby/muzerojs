import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { HiddenState, Network, Observation } from './nnet'
import {Action} from "../selfplay/mctsnode";

class UniformHiddenState implements HiddenState {
}

export class UniformNetwork implements Network {
  constructor (
    // Length of the action tensors
    private readonly actionSpace: number
  ) {}

  public initialInference (obs: Observation): NetworkOutput {
    const hiddenState = new UniformHiddenState()
    const policy = new Array<number>(this.actionSpace).fill(1 / this.actionSpace)
    return new NetworkOutput(0, 0, policy, hiddenState)
  }

  public recurrentInference (hiddenState: HiddenState, action: Action): NetworkOutput {
    const newHiddenState = new UniformHiddenState()
    const policy = new Array<number>(this.actionSpace).fill(1 / this.actionSpace)
    return new NetworkOutput(0, 0, policy, newHiddenState)
  }

  public trainInference (samples: Array<Batch>): number[] {
    // A uniform network should never be trained
    throw new Error('Training has been attempted on a uniform mocked network. This is not allowed.')
  }

  public async save (path: string): Promise<void> {
    // No reason for saving anything from a uniform network
  }

  public async load (path: string): Promise<void> {
    // We can't load any data.old to a uniform network
    throw new Error('Load weights has been attempted on a uniform mocked network. This is not allowed.')
  }

  public copyWeights (network: Network): void {
    // A uniform network does not have any data.old to copy - leave the target network untouched
  }
}
