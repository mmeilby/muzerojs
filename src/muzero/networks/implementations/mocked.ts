import * as tf from '@tensorflow/tfjs-node-gpu'
import { TensorNetworkOutput } from '../networkoutput'
import { type Batch } from '../../replaybuffer/batch'
import { type Playerwise } from '../../selfplay/entities'
import { type Network } from '../nnet'
import { type Environment } from '../../games/core/environment'
import type { Model } from '../model'

/**
 * Mocked network for MuZero reinforced learning
 */
export class MockedNetwork<State extends Playerwise> implements Network {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor (
    private readonly env: Environment<State>,
    private readonly getState: (observation: tf.Tensor) => State
  ) {
    this.actionSpaceN = env.config().actionSpace
  }

  public initialInference (observation: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with the perfect move
    const state = this.getState(observation)
    const tfValue = tf.tensor2d([[this.env.reward(state, state.player)]])
    const action = this.env.expertAction(state)
    const tfPolicy = this.policyTransform(action.id)
    return new TensorNetworkOutput(tfValue, tf.zerosLike(tfValue), tfPolicy, observation)
  }

  public recurrentInference (hiddenState: tf.Tensor, action: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with the perfect move
    const state = this.getState(hiddenState)
    const newState = this.env.step(state, action)
    const tfValue = tf.tensor2d([[this.env.reward(newState, newState.player)]])
    const newAction = this.env.expertAction(newState)
    const tfPolicy = this.policyTransform(newAction.id)
    return new TensorNetworkOutput(tfValue, tfValue, tfPolicy, newState.observation)
  }

  public trainInference (_: Batch[]): number[] {
    // Return the perfect loss and accuracy of 100%
    return [0, 1]
  }

  public getModel (): Model {
    throw new Error('A mocked network has no model to return. GetModel is not implemented.')
  }

  public async save (_: string): Promise<void> {
    // No reason for saving anything from a mocked network
  }

  public async load (_: string): Promise<void> {
    // We can't load any data.old to a mocked network - ignore
  }

  public copyWeights (_: Network): void {
    // A mocked network does not have any data to copy - leave the target network untouched
  }

  public dispose (): number {
    // Nothing to dispose for a mocked network
    return 0
  }

  private policyTransform (policy: number): tf.Tensor {
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN, 1, 0, 'float32')
  }
}
