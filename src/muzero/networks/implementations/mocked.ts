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
export class MockedNetwork implements Network {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor (
    private readonly env: Environment,
    private readonly getState: (observation: tf.Tensor) => Playerwise
  ) {
    this.actionSpaceN = env.config().actionSpace
  }

  public initialInference (observation: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with the perfect move
    const tfValues: tf.Tensor[] = []
    const tfPolicies: tf.Tensor[] = []
    for (const obs of tf.unstack(observation)) {
      const state = this.getState(obs)
      tfValues.push(tf.tensor2d([[this.env.reward(state, state.player)]]))
      tfPolicies.push(this.env.expertActionPolicy(state).expandDims(0))
    }
    const tfValue = tf.stack(tfValues)
    const tfPolicy = tf.stack(tfPolicies)
    return new TensorNetworkOutput(tfValue, tf.zerosLike(tfValue), tfPolicy, observation)
  }

  public recurrentInference (hiddenState: tf.Tensor, action: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with the perfect move
    const tfValues: tf.Tensor[] = []
    const tfPolicies: tf.Tensor[] = []
    const tfObs: tf.Tensor[] = []
    const tfActions: tf.Tensor[] = tf.unstack(action)
    tf.unstack(hiddenState).forEach((hs, index) => {
      const state = this.getState(hs)
      const newState = this.env.step(state, {id: tfActions[index].argMax().bufferSync().get(0)})
      tfValues.push(tf.tensor2d([[this.env.reward(newState, state.player)]]))
      tfPolicies.push(this.env.expertActionPolicy(newState).expandDims(0))
      tfObs.push(newState.observation)
    })
    const tfValue = tf.stack(tfValues)
    const tfPolicy = tf.stack(tfPolicies)
    return new TensorNetworkOutput(tfValue, tfValue, tfPolicy, tf.stack(tfObs))
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
}
