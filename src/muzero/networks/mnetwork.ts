import * as tf from '@tensorflow/tfjs-node'
import { NetworkOutput } from './networkoutput'
import { MuZeroBatch } from '../replaybuffer/batch'
import { Actionwise, Playerwise } from '../selfplay/entities'
import { MuZeroHiddenState, MuZeroNetwork, MuZeroObservation } from './nnet'
import { MuZeroEnvironment } from '../games/core/environment'
import {MuZeroNetObservation} from "./network";

class MuZeroMockedObservation<State> implements MuZeroObservation {
  constructor (
    public state: State
  ) {}
}

class MuZeroMockedHiddenState<State> implements MuZeroHiddenState {
  constructor (
    public state: State
  ) {}
}

export class MuZeroMockedNetwork<State extends Playerwise, Action extends Actionwise> implements MuZeroNetwork<Action> {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor (
    private readonly env: MuZeroEnvironment<State, Action>
  ) {
    this.actionSpaceN = env.config().actionSpaceSize
  }

  public initialInference (obs: MuZeroObservation): NetworkOutput {
    if (!(obs instanceof MuZeroMockedObservation<State>)) {
      throw new Error(`Incorrect observation applied to initialInference`)
    }
    // The mocked network will respond with the perfect move
    const action = this.env.expertAction(obs.state)
    const reward = this.env.reward(obs.state, obs.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(action.id)
    const hiddenState = new MuZeroMockedHiddenState(obs.state)
    const policy = tfPolicy.arraySync() as number[]
    return new NetworkOutput(value, reward, policy, hiddenState)
  }

  public recurrentInference (hiddenState: MuZeroHiddenState, action: Action): NetworkOutput {
    if (!(hiddenState instanceof MuZeroMockedHiddenState)) {
      throw new Error(`Incorrect hidden state applied to recurrentInference`)
    }
    // The mocked network will respond with the perfect move
    const newState = this.env.step(hiddenState.state, action)
    const newAction = this.env.expertAction(hiddenState.state)
    const reward = this.env.reward(hiddenState.state, hiddenState.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(newAction.id)
    const policy = tfPolicy.arraySync() as number[]
    const newHiddenState = new MuZeroMockedHiddenState(newState)
    return new NetworkOutput(value, reward, policy, newHiddenState)
  }

  public trainInference (samples: Array<MuZeroBatch<Actionwise>>): number[] {
    // Return the perfect loss and accuracy of 100%
    return [0, 1]
  }

  public async save (path: string): Promise<void> {
    // No reason for saving anything from a mocked network
  }

  public async load (path: string): Promise<void> {
    // We can't load any data to a mocked network - ignore
  }

  public copyWeights (network: MuZeroNetwork<Action>): void {
    // A mocked network does not have any data to copy - leave the target network untouched
  }

  private policyTransform (policy: number): tf.Tensor {
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN, 1, 0, 'float32')
  }
}
