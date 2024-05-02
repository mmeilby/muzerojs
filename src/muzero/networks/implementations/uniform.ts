import { type Batch } from '../../replaybuffer/batch'
import { type Network } from '../nnet'
import * as tf from '@tensorflow/tfjs-node'
import { TensorNetworkOutput } from '../networkoutput'
import type { Model } from '../model'

export class UniformNetwork implements Network {
  constructor (
    // Length of the action tensors
    private readonly actionSpace: number
  ) {
  }

  public initialInference (observation: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with a uniform distributed probability for all actions
    const tfPolicy = tf.fill([1, this.actionSpace], 1 / this.actionSpace)
    return new TensorNetworkOutput(tf.zeros([1, 1]), tf.zeros([1, 1]), tfPolicy, observation)
  }

  public recurrentInference (hiddenState: tf.Tensor, _: tf.Tensor): TensorNetworkOutput {
    // The mocked network will respond with a uniform distributed probability for all actions
    const tfPolicy = tf.fill([1, this.actionSpace], 1 / this.actionSpace)
    return new TensorNetworkOutput(tf.zeros([1, 1]), tf.zeros([1, 1]), tfPolicy, hiddenState)
  }

  public trainInference (_: Batch[]): number[] {
    // A uniform network should never be trained
    throw new Error('Training has been attempted on a uniform mocked network. This is not allowed.')
  }

  public getModel (): Model {
    throw new Error('A uniform mocked network has no model to return. GetModel is not implemented.')
  }

  public async save (_: string): Promise<void> {
    // No reason for saving anything from a uniform network
  }

  public async load (_: string): Promise<void> {
    // We can't load any data to a uniform network
    throw new Error('Load weights has been attempted on a uniform mocked network. This is not allowed.')
  }

  public copyWeights (_: Network): void {
    // A uniform network does not have any data to copy - leave the target network untouched
  }

  public dispose (): number {
    // Nothing to dispose for a uniform network
    return 0
  }
}
