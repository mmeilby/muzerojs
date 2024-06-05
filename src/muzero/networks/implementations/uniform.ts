import { type Batch } from '../../replaybuffer/batch'
import { type Network } from '../nnet'
import * as tf from '@tensorflow/tfjs-node-gpu'
import { TensorNetworkOutput } from '../networkoutput'
import type { Model } from '../model'
import { type NetworkState } from '../networkstate'
import type { Action } from '../../games/core/action'
import { type LossLog } from './core'
import { type Config } from '../../games/core/config'

export class UniformNetwork implements Network {
  constructor (
    // Length of the action tensors
    private readonly config: Config
  ) {
  }

  public initialInference (state: NetworkState): TensorNetworkOutput {
    // The mocked network will respond with a uniform distributed probability for all actions
    const tfPolicy = tf.fill([1, this.config.actionSpace], 1 / this.config.actionSpace)
    return new TensorNetworkOutput(tf.zeros([1, 1]), tf.zeros([1, 1]), tfPolicy, state.hiddenState)
  }

  public recurrentInference (state: NetworkState, action: Action[]): TensorNetworkOutput {
    // The mocked network will respond with a uniform distributed probability for all actions
    const tfPolicy = tf.fill([1, this.config.actionSpace], 1 / this.config.actionSpace)
    return new TensorNetworkOutput(tf.zeros([1, 1]), tf.zeros([1, 1]), tfPolicy, state.hiddenState)
  }

  public trainInference (_: Batch[]): LossLog {
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
