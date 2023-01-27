import { MuZeroSharedStorage } from './sharedstorage'
import { MuZeroReplayBuffer } from '../replaybuffer/replaybuffer'
import * as tf from '@tensorflow/tfjs-node'

import { Actionwise, Playerwise } from '../selfplay/entities'

import debugFactory from 'debug'
import { MuZeroBatch } from '../replaybuffer/batch'
import { BaseMuZeroNet } from '../networks/network'
const debug = debugFactory('muzero:training:module')

export class MuZeroTraining<State extends Playerwise, Action extends Actionwise> {
  // Total number of training steps (ie weights update according to a batch)
  private readonly trainingSteps: number
  // Number of training steps before using the model for self-playing
  private readonly checkpointInterval: number

  // Number of steps in the future to take into account for calculating the target value
  private readonly tdSteps: number
  // Number of game moves to keep for every batch element
  private readonly numUnrollSteps: number

  // L2 weights regularization
  private readonly weightDecay: number
  // Used only if optimizer is SGD
  private readonly momentum: number

  // Exponential learning rate schedule

  // Initial learning rate
  private readonly lrInit: number
  // Set it to 1 to use a constant learning rate
  private readonly lrDecayRate: number
  // Number of steps to decay the learning rate?
  private readonly lrDecaySteps: number

  constructor (config: {
    trainingSteps: number
    checkpointInterval: number
    tdSteps: number
    numUnrollSteps?: number
    weightDecay?: number
    learningRate?: number
  }) {
    this.trainingSteps = config.trainingSteps
    this.checkpointInterval = config.checkpointInterval
    this.tdSteps = config.tdSteps
    this.numUnrollSteps = config.numUnrollSteps ?? 10
    this.weightDecay = config.weightDecay ?? 0.0001
    this.lrInit = config.learningRate ?? 0.001
    this.lrDecayRate = 1
    this.lrDecaySteps = 10000
    this.momentum = 0.9
  }

  public async trainNetwork (storage: MuZeroSharedStorage, replayBuffer: MuZeroReplayBuffer<State, Action>): Promise<void> {
    const network = storage.uniformNetwork(this.lrInit)
    storage.latestNetwork().copyWeights(network)
    debug('Training initiated')
    debug(`Training steps: ${this.trainingSteps}`)
    const useBaseline = tf.memory().numTensors
    for (let step = 1; step <= this.trainingSteps; step++) {
      if (step % this.checkpointInterval === 0) {
        await storage.saveNetwork(step, network)
      }
      const batchSamples = replayBuffer.sampleBatch(this.numUnrollSteps, this.tdSteps).filter(batch => batch.actions.length > 0)
      const [ losses, accuracy ] = await network.trainInference(batchSamples)
      debug(`Mean loss: ${losses.toFixed(3)}, accuracy: ${accuracy.toFixed(3)}`)
      if (tf.memory().numTensors - useBaseline > 0) {
        debug(`TENSOR USAGE IS GROWING: ${tf.memory().numTensors - useBaseline}`)
      }
      await tf.nextFrame()
    }
    await storage.saveNetwork(this.trainingSteps, network)
  }

  /**
   * Discount the reward values.
   *
   * @param {number[]} rewards The reward values to be discounted.
   * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
   *   0.95.
   * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
   */
  private discountRewards (rewards: number[], discountRate: number): tf.Tensor<tf.Rank> {
    const discountedBuffer = tf.buffer([rewards.length])
    let prev = 0
    for (let i = rewards.length - 1; i >= 0; --i) {
      const current = discountRate * prev + rewards[i]
      discountedBuffer.set(current, i)
      prev = current
    }
    return discountedBuffer.toTensor()
  }

  /**
   * Discount and normalize reward values.
   *
   * This function performs two steps:
   *
   * 1. Discounts the reward values using `discountRate`.
   * 2. Normalize the reward values with the global reward mean and standard
   *    deviation.
   *
   * @param {number[][]} rewardSequences Sequences of reward values.
   * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
   *   0.95.
   * @returns {tf.Tensor[]} The discounted and normalize reward values as an
   *   Array of tf.Tensor.
   */
  private discountAndNormalizeRewards (rewardSequences: number[][], discountRate: number): Array<tf.Tensor<tf.Rank>> {
    return tf.tidy(() => {
      const discounted = []
      for (const sequence of rewardSequences) {
        discounted.push(this.discountRewards(sequence, discountRate))
      }
      // Compute the overall mean and stddev.
      const concatenated = tf.concat(discounted)
      const mean = tf.mean(concatenated)
      const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))))
      // Normalize the reward sequences using the mean and std.
      const normalized = discounted.map(rs => rs.sub(mean).div(std))
      return normalized
    })
  }

  /**
   * Scale the gradient values using normalized reward values and compute average.
   *
   * The gradient values are scaled by the normalized reward values. Then they
   * are averaged across all games and all steps.
   *
   * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
   *   name to all the gradient values for the variable across all games and all
   *   steps.
   * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
   *   for all the games. Each element of the Array is a 1D tf.Tensor of which
   *   the length equals the number of steps in the game.
   * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
   *   for the variables.
   */
  private scaleAndAverageGradients (allGradients: Record<string, tf.Tensor[][]>): Record<string, tf.Tensor<tf.Rank>> { //, normalizedRewards: tf.Tensor[]) {
    return tf.tidy(() => {
      const gradients: Record<string, tf.Tensor> = {}
      for (const varName in allGradients) {
        gradients[varName] = tf.tidy(() => {
          // Stack gradients together.
          const varGradients = allGradients[varName].map(varGameGradients => tf.stack(varGameGradients))
          // Expand dimensions of reward tensors to prepare for multiplication
          // with broadcasting.
          /*
          const expandedDims = [];
          for (let i = 0; i < varGradients[0].rank - 1; ++i) {
            expandedDims.push(1);
          }
          const reshapedNormalizedRewards = normalizedRewards.map(
            rs => rs.reshape(rs.shape.concat(expandedDims)));
          for (let g = 0; g < varGradients.length; ++g) {
            // This mul() call uses broadcasting.
            varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
          }

           */
          // Concatenate the scaled gradients together, then average them across
          // all the steps of all the games.
          return tf.concat(varGradients, 0).mean(0)
        })
      }
      return gradients
    })
  }
}
