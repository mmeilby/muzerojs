import type * as tf from '@tensorflow/tfjs-node-gpu'

export interface TrainedNetworkOutput {
  grads: tf.NamedTensorMap
  loss: tf.Scalar
  state: tf.Tensor
}
