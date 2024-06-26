import type * as tf from '@tensorflow/tfjs-node-gpu'

export class Target {
  constructor (
    // The value target is the discounted root value of the search tree N steps
    // into the future, plus the discounted sum of all rewards until then
    public readonly value: tf.Tensor,
    // The reward is the achieved score for this target state
    public readonly reward: tf.Tensor,
    // The policy represents the probability vector to most likely success
    // (Number of child visits for each action at this target state - found by MCTS)
    public readonly policy: tf.Tensor
  ) {
  }
}
