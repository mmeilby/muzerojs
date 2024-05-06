import type * as tf from '@tensorflow/tfjs-node-gpu'

export class TensorNetworkOutput {
  constructor (
    public tfValue: tf.Tensor,
    public tfReward: tf.Tensor,
    public tfPolicy: tf.Tensor,
    public tfHiddenState: tf.Tensor
  ) {
  }
}
