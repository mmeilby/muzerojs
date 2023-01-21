import * as tf from '@tensorflow/tfjs-node'

export class NetworkOutput {
  value: tf.Tensor
  nValue: number
  reward: tf.Tensor
  nReward: number
  policy: tf.Tensor // policy logits
  policyMap: number[]
  hiddenState: tf.Tensor
  aHiddenState: number[]

  constructor (
    value: tf.Tensor,
    nValue: number,
    reward: tf.Tensor,
    nReward: number,
    policy: tf.Tensor,
    policyMap: number[],
    hiddenState: tf.Tensor,
    aHiddenState: number[]
  ) {
    this.value = value
    this.nValue = nValue
    this.reward = reward
    this.nReward = nReward
    this.policy = policy
    this.policyMap = policyMap
    this.hiddenState = hiddenState
    this.aHiddenState = aHiddenState
  }
}
