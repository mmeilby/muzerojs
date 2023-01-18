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

  // Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
  private readonly valueLossWeight: number

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
    this.lrInit = config.learningRate ?? 0.1
    this.lrDecayRate = 1
    this.lrDecaySteps = 10000
    this.momentum = 0.9
    this.valueLossWeight = 0.25
  }

  public async trainNetwork (storage: MuZeroSharedStorage, replayBuffer: MuZeroReplayBuffer<State, Action>): Promise<void> {
    const network = await storage.latestNetwork()
    debug('Training initiated')
    debug('Training steps: %d', this.trainingSteps)
    let learningRate = this.lrInit
    for (let step = 1; step <= this.trainingSteps; step++) {
      if (step % this.checkpointInterval === 0) {
        await storage.saveNetwork(step, network)
      }
      tf.tidy(() => {
        const batchSamples = replayBuffer.sampleBatch(this.numUnrollSteps, this.tdSteps)
        for (const batch of batchSamples) {
          if (batch.actions.length > 0) {
            const lossFunc = (): tf.Scalar => this.calcLoss(network, batch)
            const optimizer = tf.train.momentum(learningRate, this.momentum)
            const cost = optimizer.minimize(lossFunc, true)
            if (cost !== null) {
              debug(`Cost: ${cost.bufferSync().get(0).toFixed(3)}`)
            }
          }
        }
        if (step < this.lrDecaySteps) {
          learningRate *= this.lrDecayRate
        }
      })
      await tf.nextFrame()
    }
    await storage.saveNetwork(this.trainingSteps, network)
  }

  private calcLoss (network: BaseMuZeroNet, batch: MuZeroBatch<Action>): tf.Scalar {
    const loss: tf.Scalar[] = []
    const target = batch.targets[0]
    const networkOutputInitial = network.initialInference(batch.image)
    loss.push(
      network.lossPolicy(target.policy, networkOutputInitial.policy).add(
        network.lossValue(target.value, networkOutputInitial.value))
    )
    let hiddenState: tf.Tensor = networkOutputInitial.hiddenState
    batch.actions.forEach((action, i) => {
      const target = batch.targets[i + 1]
      if (target.policy.length > 1) {
        const networkOutput = network.recurrentInference(hiddenState, network.policyTransform(action.id))
        loss.push(
          network.lossReward(target.reward, networkOutput.reward).add(
            network.lossPolicy(target.policy, networkOutput.policy)).add(
            network.lossValue(target.value, networkOutput.value))
        )
        hiddenState = networkOutput.hiddenState
      } else {
        debug(`ERROR - policy malformed: ${JSON.stringify(batch)} I=${i}`)
      }
    })
    return loss.reduce((sum, l) => sum.add(l), tf.scalar(0))
  }

  private updateWeights (network: BaseMuZeroNet, batchSamples: Array<MuZeroBatch<Action>>, learningRate: number): void {
    //    const optimizer = tf.train.momentum(learningRate, this.momentum)
    const optimizerSGD = tf.train.sgd(learningRate)
    const loss: tf.Scalar[] = []
    const lossInitialInference: tf.Scalar[] = []
    const allGradients: Record<string, tf.Tensor[][]> = {}
    for (const batch of batchSamples) {
      const gameGradients: Record<string, tf.Tensor[]> = {}
      if (batch.actions.length > 0) {
        const initialTarget = batch.targets[0]
        const gradients = network.trainInitialInference(batch.image, initialTarget.policy, initialTarget.value)
        this.pushGradients(gameGradients, gradients.grads)
        loss.push(gradients.loss)
        lossInitialInference.push(gradients.loss)
        // Initial step, from the real observation.
        let hiddenState: tf.Tensor = gradients.state
        const lossScale = 1 / batch.actions.length
        batch.actions.forEach((action, i) => {
          const target = batch.targets[i + 1]
          if (target.policy.length > 1) {
            const gradients = network.trainRecurrentInference(
              hiddenState,
              network.policyTransform(action.id),
              target.policy, target.value, target.reward,
              lossScale)
            this.pushGradients(gameGradients, gradients.grads)
            loss.push(gradients.loss)
            hiddenState = gradients.state
          } else {
            debug(`ERROR - policy malformed: ${JSON.stringify(batch)} I=${i}`)
          }
        })
      }
      // transfer all gradients
      for (const key in gameGradients) {
        if (key in allGradients) {
          allGradients[key].push(gameGradients[key])
        } else {
          allGradients[key] = [gameGradients[key]]
        }
      }
    }

    // The following line does three things:
    // 1. Performs reward discounting, i.e., make recent rewards count more
    //    than rewards from the further past. The effect is that the reward
    //    values from a game with many steps become larger than the values
    //    from a game with fewer steps.
    // 2. Normalize the rewards, i.e., subtract the global mean value of the
    //    rewards and divide the result by the global standard deviation of
    //    the rewards. Together with step 1, this makes the rewards from
    //    long-lasting games positive and rewards from short-lasting
    //    negative.
    // 3. Scale the gradients with the normalized reward values.
    //      const normalizedRewards =
    //        this.discountAndNormalizeRewards(allRewards, discountRate);
    // Add the scaled gradients to the weights of the policy network. This
    // step makes the policy network more likely to make choices that lead
    // to long-lasting games in the future (i.e., the crux of this RL
    // algorithm.)
    optimizerSGD.applyGradients(this.scaleAndAverageGradients(allGradients)) //, normalizedRewards));
    tf.dispose(allGradients)
    debug(`Initial inference loss: ${tf.mean(tf.stack(lossInitialInference)).bufferSync().get(0).toFixed(3)}, Total loss: ${tf.mean(tf.stack(loss)).bufferSync().get(0).toFixed(3)}`)
  }

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
  private pushGradients (record: Record<string, tf.Tensor[]>, gradients: Record<string, tf.Tensor>): void {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key])
      } else {
        record[key] = [gradients[key]]
      }
    }
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
