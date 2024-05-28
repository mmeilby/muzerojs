import * as tf from '@tensorflow/tfjs-node-gpu'
import { TensorNetworkOutput } from '../networkoutput'
import { type Batch } from '../../replaybuffer/batch'
import { type Network } from '../nnet'
import { type Model } from '../model'

import debugFactory from 'debug'
import { NetworkState } from '../networkstate'
import { type Action } from '../../games/core/action'
import { type Config } from '../../games/core/config'

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
  public accuracy: number

  constructor () {
    this.value = 0
    this.reward = 0
    this.policy = 0
    this.accuracy = 0
  }
}

/**
 * Core network wrapper for MuZero reinforced learning
 */
export class CoreNet implements Network {
  constructor (
    private readonly model: Model,
    private readonly config: Config
  ) {
  }

  /**
   * Predict the first state, policy, and value for the initial board observation
   * ```
   * h(o)->s
   * f(s)->p,v
   * ```
   * @param state
   */
  public initialInference (state: NetworkState): TensorNetworkOutput {
    const tfHiddenState = this.model.representation(state.hiddenState)
    const tfPolicy = this.model.policy(tfHiddenState)
    const tfValue = this.model.value(tfHiddenState)
    return new TensorNetworkOutput(tfValue, tf.zerosLike(tfValue), tfPolicy, tfHiddenState)
  }

  /*
    public async trainInitialInference (state: NetworkState, targets: Prediction): Promise<tf.Tensor> {
      const tfHiddenState = this.model.representation(state.hiddenState)
      const loss = await Promise.all([
        this.model.trainPolicy(tfHiddenState, targets.policy),
        this.model.trainValue(tfHiddenState, targets.value)
      ])
      return tfHiddenState
    }
  */
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

  /*
    public async trainRecurrentInference (state: NetworkState, action: Action[], targets: Prediction): Promise<tf.Tensor> {
      const conditionedHiddenState = tf.concat([state.hiddenState, tf.stack(action.map(a => a.action))], 1)
      const tfHiddenState = this.model.dynamics(conditionedHiddenState)
      const loss = await Promise.all([
        this.model.trainPolicy(tfHiddenState, targets.policy),
        this.model.trainValue(tfHiddenState, targets.value),
        this.model.trainReward(conditionedHiddenState, targets.reward)
      ])
      conditionedHiddenState.dispose()
      return tfHiddenState
    }
  */
  public trainInference (samples: Batch[]): number[] {
    debug(`Training sample set of ${samples.length} games`)
    const lossLog: LossLog = new LossLog()
    const optimizer = tf.train.rmsprop(this.config.lrInit, this.config.lrDecayRate, this.config.momentum)
    // const cost = optimizer.minimize(() => this.calculateLoss(samples, lossLog), true, this.model.getHiddenStateWeights())
    const cost = optimizer.minimize(() => this.calculateLoss(samples, lossLog), true)
    const loss = cost?.bufferSync().get(0) ?? 0
    optimizer.dispose()
    return [loss, lossLog.accuracy, lossLog.value, lossLog.reward, lossLog.policy]
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
   * @param batchTotalLoss
   * @returns `LossLog` record containing total loss and individual loss parts averaged for the batch
   */
  private calculateLoss (samples: Batch[], batchTotalLoss: LossLog): tf.Scalar {
    const predictions: Prediction[] = this.preparePredictions(samples)
    const labels: Prediction[] = this.prepareLabels(samples, predictions)
    const loss = {
      value: tf.scalar(0),
      reward: tf.scalar(0),
      policy: tf.scalar(0),
      total: tf.scalar(0)
    }
    const accuracy = {
      value: tf.scalar(0),
      reward: tf.scalar(0),
      policy: tf.scalar(0),
      total: tf.scalar(0)
    }
    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i]
      const label = labels[i]
      // Get the mean value loss for the batches at step i
      const lossV = tf.losses.meanSquaredError(label.value, prediction.value)
      // Get the mean value accuracy for the batches at step i
      // const accV = tf.tensor1d([1.0]).sub(tf.metrics.meanAbsolutePercentageError(label.value, prediction.value).mean())
      loss.value = loss.value.add(lossV)
      loss.total = loss.total.add(this.scaleGradient(lossV.mul(this.config.valueScale), prediction.scale))
      if (i > 0) {
        // Get the mean reward loss for the batches at step i
        const lossR = tf.losses.meanSquaredError(label.reward, prediction.reward)
        // Get the mean reward accuracy for the batches at step i
        // const accR = tf.tensor1d([1.0]).sub(tf.metrics.meanAbsolutePercentageError(label.reward, prediction.reward).mean())
        loss.reward = loss.reward.add(lossR)
        loss.total = loss.total.add(this.scaleGradient(lossR, prediction.scale))
      }
      // Get the mean policy loss for the batches at step i
      const lossP = tf.losses.softmaxCrossEntropy(label.policy, prediction.policy)
      // Get the mean policy accuracy for the batches at step i
      const accP = tf.metrics.categoricalAccuracy(label.policy, prediction.policy).mean()
      accuracy.policy = accuracy.policy.add(accP)
      loss.policy = loss.policy.add(lossP)
      loss.total = loss.total.add(this.scaleGradient(lossP, prediction.scale))
    }
    batchTotalLoss.value = loss.value.div(predictions.length).bufferSync().get(0)
    batchTotalLoss.reward = loss.reward.div(predictions.length).bufferSync().get(0)
    batchTotalLoss.policy = loss.policy.div(predictions.length).bufferSync().get(0)
    batchTotalLoss.accuracy = accuracy.policy.div(predictions.length).bufferSync().get(0)
    if (debug.enabled) {
      debug(`Sample set loss details: V=${batchTotalLoss.value.toFixed(3)} R=${batchTotalLoss.reward.toFixed(3)} P=${batchTotalLoss.policy.toFixed(3)}`)
      debug(`Sample set mean loss: T=${loss.total.bufferSync().get(0).toFixed(3)}`)
    }
    return loss.total.asScalar()
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
    for (let step = 0; step < this.config.numUnrollSteps; step++) {
      actions[step] = []
      for (let batchId = 0; batchId < sample.length; batchId++) {
        actions[step][batchId] = sample[batchId].actions[step] ?? sample[batchId].actions.at(-1)
      }
    }
    let state = tno.tfHiddenState
    for (const batchActions of actions) {
      const tno = this.recurrentInference(new NetworkState(state), batchActions)
      predictions.push({
        scale: 1 / this.config.numUnrollSteps,
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
    for (let c = 0; c <= this.config.numUnrollSteps; c++) {
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
