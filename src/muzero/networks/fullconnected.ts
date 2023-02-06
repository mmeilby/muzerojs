import { BaseMuZeroNet } from './network'
import * as tf from '@tensorflow/tfjs-node'

export class MuZeroNet extends BaseMuZeroNet {

  private makeHiddenLayer (name: string, units: number): tf.layers.Layer {
    return tf.layers.dense({
      name: name,
      units: units,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
  }

  protected h (): { sh: tf.layers.Layer, s: tf.layers.Layer } {
    const hs = tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'softsign',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      sh: this.makeHiddenLayer('representation_state_hidden', this.hiddenLayerSize),
      s: hs
    }
  }

  protected f (): { vh: tf.layers.Layer, v: tf.layers.Layer, ph: tf.layers.Layer, p: tf.layers.Layer } {
    const fv = tf.layers.dense({
      name: 'prediction_value_output',
      units: this.valueSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    const fp = tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.actionSpaceN,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      vh: this.makeHiddenLayer('prediction_value_hidden', this.hiddenLayerSize),
      v: fv,
      ph: this.makeHiddenLayer('prediction_policy_hidden', this.hiddenLayerSize),
      p: fp
    }
  }

  protected g (): { sh: tf.layers.Layer, s: tf.layers.Layer, rh: tf.layers.Layer, r: tf.layers.Layer } {
    const gs = tf.layers.dense({
      name: 'dynamics_state_output',
      units: this.hxSize,
      activation: 'softsign',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    const gr = tf.layers.dense({
      name: 'dynamics_reward_output',
      units: this.rewardSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      sh: this.makeHiddenLayer('dynamics_state_hidden', this.hiddenLayerSize),
      s: gs,
      rh: this.makeHiddenLayer('dynamics_reward_hidden', this.hiddenLayerSize),
      r: gr
    }
  }
}
