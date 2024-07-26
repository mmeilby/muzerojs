import type * as tf from '@tensorflow/tfjs-node-gpu'

export interface Model {
  representation: (observation: tf.Tensor) => tf.Tensor
  value: (state: tf.Tensor) => tf.Tensor
  policy: (state: tf.Tensor) => tf.Tensor
  dynamics: (conditionedState: tf.Tensor) => tf.Tensor
  reward: (conditionedState: tf.Tensor) => tf.Tensor
  trainPolicy: (labels: tf.Tensor, targets: tf.Tensor) => Promise<tf.History>
  trainValue: (labels: tf.Tensor, targets: tf.Tensor) => Promise<tf.History>
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Model) => void
  dispose: () => number
  print: () => void
}
