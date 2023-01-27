import { BaseMuZeroNet } from './network'
import * as tf from '@tensorflow/tfjs-node'

export class MuZeroNet extends BaseMuZeroNet {

  protected h (observationInput: tf.SymbolicTensor): { s: tf.SymbolicTensor } {
    const hs = tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
      biasRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
    }).apply(observationInput)
    return {
      s: hs as tf.SymbolicTensor,
    }
  }

  protected f (stateInput: tf.SymbolicTensor): { v: tf.SymbolicTensor, p: tf.SymbolicTensor } {
    const self = this
    function makeHiddenLayer(name: string, units: number): tf.SymbolicTensor {
      return tf.layers.dense({
        name: name,
        units: units,
        activation: 'relu',
        kernelInitializer: 'glorotUniform',
        biasInitializer: 'zeros',
        kernelRegularizer: tf.regularizers.l1l2({l1: self.weightDecay, l2: self.weightDecay}),
        biasRegularizer: tf.regularizers.l1l2({l1: self.weightDecay, l2: self.weightDecay}),
      }).apply(stateInput) as tf.SymbolicTensor
    }
    const fv = tf.layers.dense({
      name: 'prediction_value_output',
      units: this.valueSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
      biasRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
    }).apply(makeHiddenLayer('prediction_value_hidden', this.hiddenLayerSize))
    const fp = tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.actionSpaceN,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
      biasRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
    }).apply(makeHiddenLayer('prediction_policy_hidden', this.hiddenLayerSize))
    return {
      v: fv as tf.SymbolicTensor,
      p: fp as tf.SymbolicTensor
    }
  }

  protected g (actionPlaneInput: tf.SymbolicTensor): { s: tf.SymbolicTensor, r: tf.SymbolicTensor } {
    const self = this
    function makeHiddenLayer(name: string, units: number): tf.SymbolicTensor {
      return tf.layers.dense({
        name: name,
        units: units,
        activation: 'relu',
        kernelInitializer: 'glorotUniform',
        biasInitializer: 'zeros',
        kernelRegularizer: tf.regularizers.l1l2({l1: self.weightDecay, l2: self.weightDecay}),
        biasRegularizer: tf.regularizers.l1l2({l1: self.weightDecay, l2: self.weightDecay}),
      }).apply(actionPlaneInput) as tf.SymbolicTensor
    }
    const gs = tf.layers.dense({
      name: 'dynamics_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
      biasRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
    }).apply(makeHiddenLayer('dynamics_state_hidden', this.hiddenLayerSize))
    const gr = tf.layers.dense({
      name: 'dynamics_reward_output',
      units: this.rewardSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
      biasRegularizer: tf.regularizers.l1l2({l1: this.weightDecay, l2: this.weightDecay}),
    }).apply(makeHiddenLayer('dynamics_reward_hidden', this.hiddenLayerSize))
    return {
      s: gs as tf.SymbolicTensor,
      r: gr as tf.SymbolicTensor
    }
  }
}
