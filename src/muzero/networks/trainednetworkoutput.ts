import * as tf from '@tensorflow/tfjs-node'

export interface TrainedNetworkOutput {
  grads: tf.NamedTensorMap
  loss: tf.Scalar
  state: tf.Tensor
}
