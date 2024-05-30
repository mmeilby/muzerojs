import { type SharedStorage } from './sharedstorage'
import { type ReplayBuffer } from '../replaybuffer/replaybuffer'
import * as tf from '@tensorflow/tfjs-node-gpu'

import debugFactory from 'debug'
import { type Config } from '../games/core/config'

const debug = debugFactory('muzero:training:debug')
const info = debugFactory('muzero:training:info')

export class Training {
  private readonly losses: number[] = []

  constructor (
    private readonly config: Config,
    private trainingStep = 0,
    private readonly logDir = './logs/muzero'
  ) {
  }

  public async trainNetwork (storage: SharedStorage, replayBuffer: ReplayBuffer): Promise<void> {
    // Get a private copy of a new untrained network
    const network = storage.initialize()
    // Update the copy with the weights of a potentially loaded network
    storage.latestNetwork().copyWeights(network)
    debug('Training initiated')
    debug(`Training steps: ${this.config.trainingSteps} - starting iteration: ${this.trainingStep}`)
    //  Use:
    //    pip install --user tensorboard  # Unless you've already installed it.
    //    C:\Users\mm\AppData\Roaming\Python\Python311\Scripts\tensorboard.exe --logdir ./logs/muzero
    const tensorBoard = tf.node.tensorBoard(this.logDir, {
      updateFreq: 'epoch',
      histogramFreq: 0
    })
    tensorBoard.setParams({
      steps: this.config.trainingSteps,
      batchSize: this.config.batchSize
    })
    await tensorBoard.onTrainBegin()
    for (let step = 1; step <= this.config.trainingSteps; step++) {
      this.trainingStep++
      if (step % this.config.checkpointInterval === 0) {
        await storage.saveNetwork(step, network)
      }
      await tensorBoard.onEpochBegin(this.trainingStep)
      await tensorBoard.onBatchBegin(1, {
        size: this.config.batchSize
      })
      // eslint-disable-next-line @typescript-eslint/naming-convention
      const [loss, accuracy, acc_value, acc_reward, loss_value, loss_reward, loss_policy] = tf.tidy(() => {
        const batchSamples = replayBuffer.sampleBatch(this.config.numUnrollSteps, this.config.tdSteps)
        const lossLog = network.trainInference(batchSamples)
        return [lossLog.total, lossLog.accPolicy, lossLog.accValue, lossLog.accReward, lossLog.value, lossLog.reward, lossLog.policy]
      })
      this.losses.push(loss)
      debug(`Mean loss: step #${this.trainingStep} ${loss.toFixed(2)}, accuracy: ${accuracy.toFixed(2)}`)
      await tensorBoard.onBatchEnd(1, {
        loss,
        accuracy,
        acc_value,
        acc_reward,
        loss_value,
        loss_reward,
        loss_policy
      })
      await tensorBoard.onEpochEnd(this.trainingStep, {
        loss,
        accuracy,
        acc_value,
        acc_reward,
        loss_value,
        loss_reward,
        loss_policy
      })
      const performance = replayBuffer.statistics()
      // Log the performance measured by number of wins by player 1 in 100 games
      tf.node.summaryFileWriter(this.logDir).scalar('perf', performance, this.trainingStep)
      // Log the current use of tensors. The expected use should include the tensors saved as game history in replay buffer
      tf.node.summaryFileWriter(this.logDir).scalar('use', tf.memory().numTensors, this.trainingStep)
      if (info.enabled) {
        info(`--- Performance: ${performance.toFixed(1)}%`)
        info(`--- Accuracy (${this.trainingStep}): ${this.statistics().toFixed(2)}`)
        info(`--- Tensor usage: ${tf.memory().numTensors.toFixed(0)}`)
      }
      await tf.nextFrame()
    }
    await tensorBoard.onTrainEnd()
    await storage.saveNetwork(this.config.trainingSteps, network)
  }

  public statistics (): number {
    const mLoss = this.losses.slice(-100)
    const mlossSum = mLoss.reduce((s, l) => s + l, 0)
    return mlossSum / mLoss.length
  }
}
