import * as tf from '@tensorflow/tfjs-node-gpu'
import {TensorNetworkOutput} from '../networkoutput'
import {type Batch} from '../../replaybuffer/batch'
import {type Network} from '../nnet'
import {type Target} from '../../replaybuffer/target'
import {type Model} from '../model'

import debugFactory from 'debug'

const debug = debugFactory('muzero:network:core')

class Prediction {
  constructor(
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

  constructor() {
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
  constructor(
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
  public initialInference(observation: tf.Tensor): TensorNetworkOutput {
    const tfHiddenState = this.model.representation(observation)
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
   * @param hiddenState
   * @param action
   */
  public recurrentInference(hiddenState: tf.Tensor, action: tf.Tensor): TensorNetworkOutput {
    const conditionedHiddenState = tf.concat([hiddenState, action], 1)
    const tfHiddenState = this.model.dynamics(conditionedHiddenState)
    const tfReward = this.model.reward(conditionedHiddenState)
    const tfPolicy = this.model.policy(tfHiddenState)
    const tfValue = this.model.value(tfHiddenState)
    conditionedHiddenState.dispose()
    return new TensorNetworkOutput(tfValue, tfReward, tfPolicy, tfHiddenState)
  }

  public trainInference(samples: Batch[]): number[] {
    debug(`Training sample set of ${samples.length} games`)
    const optimizer = tf.train.rmsprop(this.learningRate, 0.0001, 0.9)
    const cost = optimizer.minimize(() => this.calculateLoss(samples), true)
    const loss = cost?.bufferSync().get(0) ?? 0
    optimizer.dispose()
    return [loss, 0]
  }

  public getModel(): Model {
    return this.model
  }

  public async save(path: string): Promise<void> {
    await this.model.save(path)
  }

  public async load(path: string): Promise<void> {
    await this.model.load(path)
  }

  public copyWeights(network: Network): void {
    this.model.copyWeights(network.getModel())
  }

  public dispose(): number {
    return this.model.dispose()
  }

  /**
   * Get predicted values from the network for the batch
   * @param batch a game play recorded as observation image for the initial state and the following targets (policy, reward, and value) for each action taken
   * @returns array of `Prediction` used for measuring how close the network predicts the targets
   */
  private calculatePredictions(batch: Batch): Prediction[] {
    const tno = this.initialInference(batch.image.expandDims(0))
    const predictions: Prediction[] = [{
      scale: 1,
      value: tno.tfValue,
      reward: tno.tfReward,
      policy: tno.tfPolicy
    }]
    let state = tno.tfHiddenState
    for (const action of batch.tfActions) {
      const tno = this.recurrentInference(state, action)
      predictions.push({
        scale: 1 / batch.tfActions.length,
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
   * Measure the total loss for a batch
   * ```
   * Note: for value loss use MSE in board games, cross entropy between categorical values in Atari
   * ```
   * @param predictions array of predictions as tensors for this batch
   * @param targets array of targets as tensors for this batch
   * @returns `LossLog` record containing total loss and individual loss parts averaged for the batch
   */
  private measureLoss(predictions: Prediction[], targets: Target[]): LossLog {
    const batchTotalLoss: LossLog = new LossLog()
    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i]
      const target = targets[i]
      const lossV = tf.losses.meanSquaredError(target.value, prediction.value).asScalar()
      batchTotalLoss.value += lossV.bufferSync().get(0)
      batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossV.mul(this.valueScale), prediction.scale))
      if (i > 0) {
        const lossR = tf.losses.meanSquaredError(target.reward, prediction.reward).asScalar()
        batchTotalLoss.reward += lossR.bufferSync().get(0)
        batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossR, prediction.scale))
      }
      if ((target.policy.shape.at(1) ?? 0) > 0) {
        const lossP = tf.losses.softmaxCrossEntropy(target.policy, prediction.policy).asScalar()
        batchTotalLoss.policy += lossP.bufferSync().get(0)
        batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossP, prediction.scale))
      }
    }
    batchTotalLoss.value /= predictions.length
    batchTotalLoss.reward /= predictions.length
    batchTotalLoss.policy /= predictions.length
    return batchTotalLoss
  }

  private calculateLoss(samples: Batch[]): tf.Scalar {
    const batchLosses: LossLog = this.calculateSampleLoss(samples)
    /*
            for (const batch of samples) {
                const predictions = this.calculatePredictions(batch)
                const lossAndGradients = this.measureLoss(predictions, batch.targets)
                batchLosses.total = batchLosses.total.add(lossAndGradients.total)
                if (debug.enabled) {
                    const lossV = lossAndGradients.value.toFixed(3)
                    const lossR = lossAndGradients.reward.toFixed(3)
                    const lossP = lossAndGradients.policy.toFixed(3)
                    const total = lossAndGradients.total.bufferSync().get(0).toFixed(3)
                    debug(`Game overall loss: V=${lossV}, R=${lossR}, P=${lossP} T=${total}`)
                }
                lossAndGradients.total.dispose()
            }

     */
    batchLosses.total = batchLosses.total.div(samples.length)
    if (debug.enabled) {
      debug(`Sample set loss details: V=${batchLosses.value.toFixed(3)} R=${batchLosses.reward.toFixed(3)} P=${batchLosses.policy.toFixed(3)}`)
      debug(`Sample set mean loss: T=${batchLosses.total.bufferSync().get(0).toFixed(3)}`)
    }
    // update weights
    return batchLosses.total.asScalar()
  }

  private preparePredictions(sample: Batch[]): Prediction[] {
    const images = tf.concat(sample.map(batch => batch.image.expandDims(0)))
    const tno = this.initialInference(images)
    const predictions: Prediction[] = [{
      scale: 1,
      value: tno.tfValue,
      reward: tno.tfReward,
      policy: tno.tfPolicy
    }]
    const stackedActions = sample.map(batch => tf.stack(batch.tfActions))
    const actions = tf.unstack(tf.stack(stackedActions, 1))
    let state = tno.tfHiddenState
    for (const action of actions) {
      const tno = this.recurrentInference(state, action)
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

  private prepareLabels(sample: Batch[], predictions: Prediction[]): Prediction[] {
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

  private calculateSampleLoss(sample: Batch[]): LossLog {
    const predictions: Prediction[] = this.preparePredictions(sample)
    const labels: Prediction[] = this.prepareLabels(sample, predictions)
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
      batchTotalLoss.policy += lossP.bufferSync().get(0)
      batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossP, prediction.scale))
    }
    batchTotalLoss.value /= predictions.length
    batchTotalLoss.reward /= predictions.length
    batchTotalLoss.policy /= predictions.length
    return batchTotalLoss
  }

  /**
   * Scales the gradient for the backward pass
   * @param tensor
   * @param scale
   * @private
   */
  private scaleGradient(tensor: tf.Tensor, scale: number): tf.Tensor {
    // Perform the operation: tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
    return tf.tidy(() => {
      const tidyTensor = tf.variable(tensor, false)
      const scaledGradient = tensor.mul(scale).add(tidyTensor.mul(1 - scale))
      tidyTensor.dispose()
      return scaledGradient
    })
  }
}
