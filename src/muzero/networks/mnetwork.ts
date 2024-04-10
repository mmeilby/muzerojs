import * as tf from '@tensorflow/tfjs-node'
import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { Playerwise } from '../selfplay/entities'
import { HiddenState, Network, Observation } from './nnet'
import { Environment } from '../games/core/environment'
import {Action} from "../selfplay/mctsnode";

class MockedObservation<State> implements Observation {
  constructor (
    public state: State
  ) {}
}

class MockedHiddenState<State> implements HiddenState {
  constructor (
    public state: State
  ) {}
}

export class MockedNetwork<State extends Playerwise> implements Network {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor (
    private readonly env: Environment<State>
  ) {
    this.actionSpaceN = env.config().actionSpace
  }

  public initialInference (obs: Observation): NetworkOutput {
    if (!(obs instanceof MockedObservation<State>)) {
      throw new Error(`Incorrect observation applied to initialInference`)
    }
    // The mocked network will respond with the perfect move
    const action = this.env.expertAction(obs.state)
    const reward = this.env.reward(obs.state, obs.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(action.id)
    const hiddenState = new MockedHiddenState(obs.state)
    const policy = tfPolicy.arraySync() as number[]
    return new NetworkOutput(value, reward, policy, hiddenState)
  }

  public recurrentInference (hiddenState: HiddenState, action: Action): NetworkOutput {
    if (!(hiddenState instanceof MockedHiddenState)) {
      throw new Error(`Incorrect hidden state applied to recurrentInference`)
    }
    // The mocked network will respond with the perfect move
    const newState = this.env.step(hiddenState.state, action)
    const newAction = this.env.expertAction(hiddenState.state)
    const reward = this.env.reward(hiddenState.state, hiddenState.state.player)
    const value = reward
    const tfPolicy = this.policyTransform(newAction.id)
    const policy = tfPolicy.arraySync() as number[]
    const newHiddenState = new MockedHiddenState(newState)
    return new NetworkOutput(value, reward, policy, newHiddenState)
  }

  public trainInference (samples: Array<Batch>): number[] {
    // Return the perfect loss and accuracy of 100%
    return [0, 1]
  }

  public async save (path: string): Promise<void> {
    // No reason for saving anything from a mocked network
  }

  public async load (path: string): Promise<void> {
    // We can't load any data.old to a mocked network - ignore
  }

  public copyWeights (network: Network): void {
    // A mocked network does not have any data.old to copy - leave the target network untouched
  }

  private policyTransform (policy: number): tf.Tensor {
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN, 1, 0, 'float32')
  }
}
