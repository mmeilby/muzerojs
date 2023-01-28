import { BaseMuZeroNet } from './network'
import * as tf from '@tensorflow/tfjs-node'

export class MuZeroNet extends BaseMuZeroNet {

  private makeHiddenLayer (name: string, units: number, input: tf.SymbolicTensor): tf.SymbolicTensor {
    return tf.layers.dense({
      name: name,
      units: units,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(input) as tf.SymbolicTensor
  }

  protected h (observationInput: tf.SymbolicTensor): { s: tf.SymbolicTensor } {
    const hs = tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(this.makeHiddenLayer('representation_state_hidden', this.hiddenLayerSize, observationInput))
    return {
      s: hs as tf.SymbolicTensor,
    }
  }

  protected f (stateInput: tf.SymbolicTensor): { v: tf.SymbolicTensor, p: tf.SymbolicTensor } {
    const fv = tf.layers.dense({
      name: 'prediction_value_output',
      units: this.valueSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(this.makeHiddenLayer('prediction_value_hidden', this.hiddenLayerSize, stateInput))
    const fp = tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.actionSpaceN,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(this.makeHiddenLayer('prediction_policy_hidden', this.hiddenLayerSize, stateInput))
    return {
      v: fv as tf.SymbolicTensor,
      p: fp as tf.SymbolicTensor
    }
  }

  protected g (actionPlaneInput: tf.SymbolicTensor): { s: tf.SymbolicTensor, r: tf.SymbolicTensor } {
    const gs = tf.layers.dense({
      name: 'dynamics_state_output',
      units: this.hxSize,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(this.makeHiddenLayer('dynamics_state_hidden', this.hiddenLayerSize, actionPlaneInput))
    const gr = tf.layers.dense({
      name: 'dynamics_reward_output',
      units: this.rewardSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }).apply(this.makeHiddenLayer('dynamics_reward_hidden', this.hiddenLayerSize, actionPlaneInput))
    return {
      s: gs as tf.SymbolicTensor,
      r: gr as tf.SymbolicTensor
    }
  }
}
