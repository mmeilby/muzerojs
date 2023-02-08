import * as tf from '@tensorflow/tfjs-node'
import {NetworkOutput} from "./networkoutput";
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise, Playerwise} from "../selfplay/entities";
import {MuZeroHiddenState, MuZeroNetwork, MuZeroObservation} from "./nnet";
import {MuZeroEnvironment} from "../games/core/environment";

class MuZeroMockedObservation<State> implements MuZeroObservation {
  constructor(
      public state: State
  ) {}
}

class MuZeroMockedHiddenState<State> implements MuZeroHiddenState {
  constructor(
      public state: State
  ) {}
}

export class MuZeroMockedNetwork<State extends Playerwise, Action extends Actionwise> implements MuZeroNetwork<Action> {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor(
      private readonly env: MuZeroEnvironment<State, Action>,
  ) {
    this.actionSpaceN = env.config().actionSpaceSize
  }

  public initialInference (obs: MuZeroMockedObservation<State>): NetworkOutput {
    const action = this.env.expertAction(obs.state)
    const reward = this.env.reward(obs.state, obs.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(action.id)
    const hiddenState = new MuZeroMockedHiddenState(obs.state)
    const policy = tfPolicy.arraySync() as number[]
    return new NetworkOutput(value, reward, policy, hiddenState)
  }

  public recurrentInference (hiddenState: MuZeroMockedHiddenState<State>, action: Action): NetworkOutput {
    const newState = this.env.step(hiddenState.state, action)
    const newAction = this.env.expertAction(hiddenState.state)
    const reward = this.env.reward(hiddenState.state, hiddenState.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(newAction.id)
    const policy = tfPolicy.arraySync() as number[]
    const newHiddenState = new MuZeroMockedHiddenState(newState)
    return new NetworkOutput(value, reward, policy, newHiddenState)
  }

  public async trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number[]> {
    return Promise.resolve(samples.map(() => 0))
  }
  public policyTransform (policy: number): tf.Tensor {
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN, 1, 0, 'float32')
  }
  public async save (path: string): Promise<void> {

  }
  public async load (path: string): Promise<void> {

  }
  public copyWeights (network: MuZeroNetwork<Action>): void {

  }
}
