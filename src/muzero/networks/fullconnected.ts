import { BaseMuZeroNet } from './network'
import * as tf from '@tensorflow/tfjs-node'

export class MuZeroNet extends BaseMuZeroNet {
  protected h (observationInput: tf.SymbolicTensor): tf.SymbolicTensor {
    return tf.layers.dense({
      name: 'encoded_rep_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    }).apply(observationInput) as tf.SymbolicTensor
  }

  protected f (stateInput: tf.SymbolicTensor): { v: tf.SymbolicTensor, p: tf.SymbolicTensor } {
    const f = tf.layers.dense({
      name: 'encoded_pre_state_hidden',
      units: 64,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    }).apply(stateInput) as tf.SymbolicTensor
    const encodedValueOutput = tf.layers.dense({
      name: 'encoded_value_output',
      units: this.valueSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    })
    const encodedPolicyOutput = tf.layers.dense({
      name: 'encoded_policy_output',
      units: this.actionSpaceN,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    })
    return {
      v: encodedValueOutput.apply(f) as tf.SymbolicTensor,
      p: encodedPolicyOutput.apply(f) as tf.SymbolicTensor
    }
  }

  protected g (actionPlaneInput: tf.SymbolicTensor): { s: tf.SymbolicTensor, r: tf.SymbolicTensor } {
    const g = tf.layers.dense({
      name: 'encoded_dyn_state_hidden',
      units: 64,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    }).apply(actionPlaneInput) as tf.SymbolicTensor
    const encodedDynStateOutput = tf.layers.dense({
      name: 'encoded_dyn_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    })
    const encodedRewardOutput = tf.layers.dense({
      name: 'encoded_reward_output',
      units: this.rewardSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
      kernelRegularizer: 'l1l2',
      biasRegularizer: 'l1l2'
    })
    return {
      s: encodedDynStateOutput.apply(g) as tf.SymbolicTensor,
      r: encodedRewardOutput.apply(g) as tf.SymbolicTensor
    }
  }
}
