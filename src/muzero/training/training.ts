import { type SharedStorage } from './sharedstorage'
import { type ReplayBuffer } from '../replaybuffer/replaybuffer'
import * as tf from '@tensorflow/tfjs-node-gpu'

import debugFactory from 'debug'
import { type Config } from '../games/core/config'

const debug = debugFactory('muzero:training:debug')
const info = debugFactory('muzero:training:info')

export class Training {
  private trainingStep = 0
  private readonly losses: number[] = []

  constructor (
    private readonly config: Config,
    private readonly logDir = './logs/muzero'
  ) {
  }

  public async trainNetwork (storage: SharedStorage, replayBuffer: ReplayBuffer): Promise<void> {
    // Get a private copy of a new untrained network
    const network = storage.initialize()
    // Update the copy with the weights of a potentially loaded network
    storage.latestNetwork().copyWeights(network)
    debug('Training initiated')
    debug(`Training steps: ${this.config.trainingSteps}`)
    //  Use:
    //    pip install tensorboard  # Unless you've already installed it.
    //    C:\Users\Morten\AppData\Local\Programs\Python\Python39\Scripts\tensorboard.exe --logdir ./logs/muzero
    const tensorBoard = tf.node.tensorBoard(this.logDir, {
      updateFreq: 'batch',
      histogramFreq: 0
    })
    tensorBoard.params = {
      steps: this.config.trainingSteps,
      batchSize: this.config.batchSize
    }
    await tensorBoard.onTrainBegin()
    for (let step = 1; step <= this.config.trainingSteps; step++) {
      if (step % this.config.checkpointInterval === 0) {
        await storage.saveNetwork(step, network)
      }
      await tensorBoard.onBatchBegin(step)
      const [loss, accuracy] = tf.tidy(() => {
        const batchSamples = replayBuffer.sampleBatch(this.config.numUnrollSteps, this.config.tdSteps)
        return network.trainInference(batchSamples)
      })
      this.losses.push(loss)
      debug(`Mean loss: step #${step} ${loss.toFixed(2)}, accuracy: ${accuracy.toFixed(2)}`)
      await tensorBoard.onBatchEnd(step, {val_loss: loss, val_accuracy: accuracy})
      this.trainingStep++
      if (info.enabled) {
        info(`--- Performance: ${replayBuffer.statistics().toFixed(1)}%`)
        info(`--- Accuracy (${this.trainingStep}): ${this.statistics().toFixed(2)}`)
        info(`--- Tensor usage: ${tf.memory().numTensors.toFixed(0)}`)
      }
      await tf.nextFrame()
    }
//    tf.node.summaryFileWriter(this.logDir).scalar('loss', history.history.loss[0] as number, ++this.trainingStep)
    await tensorBoard.onTrainEnd()
    await storage.saveNetwork(this.config.trainingSteps, network)
  }

  /*
  public async trainSingleInteration (storage: SharedStorage, replayBuffer: ReplayBuffer<State>): Promise<void> {
    const network = storage.latestNetwork()
    tf.tidy(() => {
      const batchSamples = replayBuffer.sampleBatch(this.config.numUnrollSteps, this.config.tdSteps).filter(batch => batch.tfActions.length > 0)
      //     debug(JSON.stringify(batchSamples))
      const [loss, accuracy] = network.trainInference(batchSamples)
      debug(`Mean loss: step #${this.trainingStep + 1} ${loss.toFixed(3)}, accuracy: ${accuracy.toFixed(3)}`)
      this.losses.push(loss)
    })
    this.trainingStep++
    if (this.trainingStep % this.config.checkpointInterval === 0) {
      await storage.saveNetwork(this.trainingStep, network)
    }
    //    (network as MuZeroNet).dispose()
  }
*/
  public statistics (): number {
    const mLoss = this.losses.slice(-100)
    const mlossSum = mLoss.reduce((s, l) => s + l, 0)
    return mlossSum / mLoss.length
  }

  /**
   * Discount the reward values.
   *
   * @param {number[]} rewards The reward values to be discounted.
   * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
   *   0.95.
   * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
   */
  /*
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
*/
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
  /*
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
      return discounted.map(rs => rs.sub(mean).div(std))
    })
  }
*/
  /**
   * Scale the gradient values using normalized reward values and compute average.
   *
   * The gradient values are scaled by the normalized reward values. Then they
   * are averaged across all games and all steps.
   *
   * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
   *   name to all the gradient values for the variable across all games and all
   *   steps.
   * param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
   *   for all the games. Each element of the Array is a 1D tf.Tensor of which
   *   the length equals the number of steps in the game.
   * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
   *   for the variables.
   */
  /*
  private scaleAndAverageGradients (allGradients: Record<string, tf.Tensor[][]>): Record<string, tf.Tensor<tf.Rank>> { //, normalizedRewards: tf.Tensor[]) {
    return tf.tidy(() => {
      const gradients: Record<string, tf.Tensor> = {}
      for (const varName in allGradients) {
        gradients[varName] = tf.tidy(() => {
          // Stack gradients together.
          const varGradients = allGradients[varName].map(varGameGradients => tf.stack(varGameGradients))
          // Expand dimensions of reward tensors to prepare for multiplication
          // with broadcasting.

//          const expandedDims = [];
//          for (let i = 0; i < varGradients[0].rank - 1; ++i) {
//            expandedDims.push(1);
//          }
//          const reshapedNormalizedRewards = normalizedRewards.map(
//            rs => rs.reshape(rs.shape.concat(expandedDims)));
//          for (let g = 0; g < varGradients.length; ++g) {
//            // This mul() call uses broadcasting.
//            varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
//          }

          // Concatenate the scaled gradients together, then average them across
          // all the steps of all the games.
          return tf.concat(varGradients, 0).mean(0)
        })
      }
      return gradients
    })
  }
*/
}
