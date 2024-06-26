import type * as tf from '@tensorflow/tfjs-node-gpu'
import { type State } from '../games/core/state'

export class TensorNetworkOutput {
  public state: State[] | undefined

  constructor (
    public tfValue: tf.Tensor,
    public tfReward: tf.Tensor,
    public tfPolicy: tf.Tensor,
    public tfHiddenState: tf.Tensor
  ) {
  }

  /**
   * Get the policy as a number array
   * The network result is assumed to be a single batch
   */
  get policy (): number[] {
    return this.tfPolicy.squeeze().arraySync() as number[]
  }

  get reward (): number {
    return this.tfReward.squeeze().bufferSync().get(0)
  }

  get value (): number {
    return this.tfValue.squeeze().bufferSync().get(0)
  }
}
