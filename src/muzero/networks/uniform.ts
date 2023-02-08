import {NetworkOutput} from "./networkoutput";
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise, Playerwise} from "../selfplay/entities";
import {MuZeroHiddenState, MuZeroNetwork, MuZeroObservation} from "./nnet";

class MuZeroUniformHiddenState implements MuZeroHiddenState {
}

export class MuZeroUniformNetwork<State extends Playerwise, Action extends Actionwise> implements MuZeroNetwork<Action> {
  constructor(
      // Length of the action tensors
      private readonly actionSpace: number,
  ) {}

  public initialInference (obs: MuZeroObservation): NetworkOutput {
    const hiddenState = new MuZeroUniformHiddenState()
    const policy = new Array<number>(this.actionSpace).fill(1/this.actionSpace)
    return new NetworkOutput(0, 0, policy, hiddenState)
  }

  public recurrentInference (hiddenState: MuZeroHiddenState, action: Action): NetworkOutput {
    const newHiddenState = new MuZeroUniformHiddenState()
    const policy = new Array<number>(this.actionSpace).fill(1/this.actionSpace)
    return new NetworkOutput(0, 0, policy, newHiddenState)
  }

  public async trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number[]> {
    // A uniform network should never be trained
    throw new Error(`Training has been attempted on a uniform mocked network. This is not allowed.`)
  }
  public async save (path: string): Promise<void> {
    // No reason for saving anything from a uniform network
  }
  public async load (path: string): Promise<void> {
    // We can't load any data to a uniform network
    throw new Error(`Load weights has been attempted on a uniform mocked network. This is not allowed.`)
  }
  public copyWeights (network: MuZeroNetwork<Action>): void {
    // A uniform network does not have any data to copy - leave the target network untouched
  }
}
