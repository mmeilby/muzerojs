import type * as tf from '@tensorflow/tfjs-node-gpu'

export interface Model {
  representation: (observation: tf.Tensor) => tf.Tensor
  value: (state: tf.Tensor) => tf.Tensor
  policy: (state: tf.Tensor) => tf.Tensor
  dynamics: (conditionedState: tf.Tensor) => tf.Tensor
  reward: (conditionedState: tf.Tensor) => tf.Tensor
  // trainRepresentation: (labels: tf.Tensor, targets: tf.Tensor) => Promise<number | number[]>
  // trainPolicy: (labels: tf.Tensor, targets: tf.Tensor) => Promise<number | number[]>
  // trainValue: (labels: tf.Tensor, targets: tf.Tensor) => Promise<number | number[]>
  // trainDynamics: (labels: tf.Tensor, targets: tf.Tensor) => Promise<number | number[]>
  // trainReward: (labels: tf.Tensor, targets: tf.Tensor) => Promise<number | number[]>
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Model) => void
  // getHiddenStateWeights: () => tf.Variable[]
  dispose: () => number
}
