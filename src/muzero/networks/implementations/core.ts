import * as tf from '@tensorflow/tfjs-node-gpu'
import { TensorNetworkOutput } from '../networkoutput'
import { type Batch } from '../../replaybuffer/batch'
import { type Network } from '../nnet'
import { type Model } from '../model'

import debugFactory from 'debug'
import { NetworkState } from '../networkstate'
import { type Action } from '../../games/core/action'

const debug = debugFactory('muzero:network:core')

class Prediction {
  constructor (
    public scale: number,
    public value: tf.Tensor,
    public reward: tf.Tensor,
    public policy: tf.Tensor
  ) {
  }
}

class LossLog {
  public value: number
  public reward: number
  public policy: number
  public total: tf.Tensor

  constructor () {
    this.value = 0
    this.reward = 0
    this.policy = 0
    this.total = tf.scalar(0)
  }
}

/**
 * Core network wrapper for MuZero reinforced learning
 */
export class CoreNet implements Network {
  constructor (
    private readonly model: Model,
    // Learning rate for SGD
    private readonly learningRate: number,
    private readonly numUnrollSteps: number,
    // Scale the value loss to avoid over fitting of the value function,
    // paper recommends 0.25 (See paper appendix Reanalyze)
    private readonly valueScale: number = 0.25
  ) {
  }

  /**
   * Predict the first state, policy, and value for the initial board observation
   * ```
   * h(o)->s
   * f(s)->p,v
   * ```
   * @param observation
   */
  public initialInference (state: NetworkState): TensorNetworkOutput {
    const tfHiddenState = this.model.representation(state.hiddenState)
    const tfPolicy = this.model.policy(tfHiddenState)
    const tfValue = this.model.value(tfHiddenState)
    return new TensorNetworkOutput(tfValue, tf.zerosLike(tfValue), tfPolicy, tfHiddenState)
  }

  /**
   * Predict the next state and reward based on current state.
   * Also predict the policy and value for the new state
   * ```
   * g(s,a)->s´,r
   * f(s´)->p,v
   * ```
   * @param state
   * @param action
   */
  public recurrentInference (state: NetworkState, action: Action[]): TensorNetworkOutput {
    const conditionedHiddenState = tf.concat([state.hiddenState, tf.stack(action.map(a => a.action))], 1)
    const tfHiddenState = this.model.dynamics(conditionedHiddenState)
    const tfReward = this.model.reward(conditionedHiddenState)
    const tfPolicy = this.model.policy(tfHiddenState)
    const tfValue = this.model.value(tfHiddenState)
    conditionedHiddenState.dispose()
    return new TensorNetworkOutput(tfValue, tfReward, tfPolicy, tfHiddenState)
  }

  public trainInference (samples: Batch[]): number[] {
    debug(`Training sample set of ${samples.length} games`)
    const optimizer = tf.train.rmsprop(this.learningRate, 0.0001, 0.9)
    const cost = optimizer.minimize(() => this.calculateLoss(samples), true)
    const loss = cost?.bufferSync().get(0) ?? 0
    optimizer.dispose()
    return [loss, 0]
  }

  public getModel (): Model {
    return this.model
  }

  public async save (path: string): Promise<void> {
    await this.model.save(path)
  }

  public async load (path: string): Promise<void> {
    await this.model.load(path)
  }

  public copyWeights (network: Network): void {
    this.model.copyWeights(network.getModel())
  }

  public dispose (): number {
    return this.model.dispose()
  }

  /**
   * Measure the total loss for a batch
   * ```
   * Note: for value loss use MSE in board games, cross entropy between categorical values in Atari
   * ```
   * @param samples
   * @returns `LossLog` record containing total loss and individual loss parts averaged for the batch
   */
  private calculateLoss (samples: Batch[]): tf.Scalar {
    const predictions: Prediction[] = this.preparePredictions(samples)
    const labels: Prediction[] = this.prepareLabels(samples, predictions)
    const batchTotalLoss: LossLog = new LossLog()
    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i]
      const label = labels[i]
      const lossV = tf.losses.meanSquaredError(label.value, prediction.value).asScalar()
      batchTotalLoss.value += lossV.bufferSync().get(0)
      batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossV.mul(this.valueScale), prediction.scale))
      if (i > 0) {
        const lossR = tf.losses.meanSquaredError(label.reward, prediction.reward).asScalar()
        batchTotalLoss.reward += lossR.bufferSync().get(0)
        batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossR, prediction.scale))
      }
      const lossP = tf.losses.softmaxCrossEntropy(label.policy, prediction.policy).asScalar()
      const acc = tf.metrics.categoricalAccuracy(label.policy, prediction.policy)
      batchTotalLoss.policy += lossP.bufferSync().get(0)
      batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossP, prediction.scale))
    }
    batchTotalLoss.value /= predictions.length
    batchTotalLoss.reward /= predictions.length
    batchTotalLoss.policy /= predictions.length
    batchTotalLoss.total = batchTotalLoss.total.div(samples.length)
    if (debug.enabled) {
      debug(`Sample set loss details: V=${batchTotalLoss.value.toFixed(3)} R=${batchTotalLoss.reward.toFixed(3)} P=${batchTotalLoss.policy.toFixed(3)}`)
      debug(`Sample set mean loss: T=${batchTotalLoss.total.bufferSync().get(0).toFixed(3)}`)
    }
    return batchTotalLoss.total.asScalar()
  }

  /**
   * Get predicted values from the network for the batch
   * @param sample a game play recorded as observation image for the initial state and the following
   * targets (policy, reward, and value) for each action taken
   * @returns array of `Prediction` used for measuring how close the network predicts the targets
   */
  private preparePredictions (sample: Batch[]): Prediction[] {
    const images = tf.concat(sample.map(batch => batch.image))
    const tno = this.initialInference(new NetworkState(images))
    const predictions: Prediction[] = [{
      scale: 1,
      value: tno.tfValue,
      reward: tno.tfReward,
      policy: tno.tfPolicy
    }]
    // Transpose the actions to align all batch actions for the same unroll step
    // If actions are missing in a batch repeat the last action to fill up to number of unrolled steps
    const actions: Action[][] = []
    for (let step = 0; step < this.numUnrollSteps; step++) {
      actions[step] = []
      for (let batchId = 0; batchId < sample.length; batchId++) {
        actions[step][batchId] = sample[batchId].actions[step] ?? sample[batchId].actions.at(-1)
      }
    }
    let state = tno.tfHiddenState
    for (const batchActions of actions) {
      const tno = this.recurrentInference(new NetworkState(state), batchActions)
      predictions.push({
        scale: 1 / this.numUnrollSteps,
        value: tno.tfValue,
        reward: tno.tfReward,
        policy: tno.tfPolicy
      })
      // Prepare new state for next game step
      // Gradient scaling controls the dynamics network training. To prevent training set scale = 0
      state = this.scaleGradient(tno.tfHiddenState, 0.5)
    }
    return predictions
  }

  /**
   * Prepare the target values (labels) for the samples
   * @param sample The samples containing the target values for each batch
   * @param predictions Array of predictions as tensors for the samples
   * @returns Array of `Prediction` records containing the target values (labels) for the training
   */
  private prepareLabels (sample: Batch[], predictions: Prediction[]): Prediction[] {
    const labels: Prediction[] = []
    for (let c = 0; c <= this.numUnrollSteps; c++) {
      labels.push({
        scale: 0,
        value: tf.concat(sample.map(batch => batch.targets[c].value)),
        reward: tf.concat(sample.map(batch => batch.targets[c].reward)),
        policy: tf.concat(sample.map((batch, i) =>
          c < batch.actions.length ? batch.targets[c].policy : tf.unstack(predictions[c].policy)[i].expandDims(0)))
      })
    }
    return labels
  }

  /**
   * Scales the gradient for the backward pass
   * @param tensor
   * @param scale
   * @private
   */
  private scaleGradient (tensor: tf.Tensor, scale: number): tf.Tensor {
    // Perform the operation: tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
    return tf.tidy(() => {
      const tidyTensor = tf.variable(tensor, false)
      const scaledGradient = tensor.mul(scale).add(tidyTensor.mul(1 - scale))
      tidyTensor.dispose()
      return scaledGradient
    })
  }
}
