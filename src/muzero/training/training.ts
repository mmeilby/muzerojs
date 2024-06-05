import { type SharedStorage } from './sharedstorage'
import { type ReplayBuffer } from '../replaybuffer/replaybuffer'
import * as tf from '@tensorflow/tfjs-node-gpu'

import debugFactory from 'debug'
import { type Config } from '../games/core/config'
import { type SummaryFileWriter } from '@tensorflow/tfjs-node-gpu/dist/tensorboard'
import { type TensorBoardCallback } from '@tensorflow/tfjs-node-gpu/dist/callbacks'
import { NetworkState } from '../networks/networkstate'

const debug = debugFactory('muzero:training:debug')
const info = debugFactory('muzero:training:info')

export class Training {
  private readonly losses: number[] = []
  private readonly summary: SummaryFileWriter
  private readonly tensorBoard: TensorBoardCallback

  constructor (
    private readonly config: Config,
    private trainingStep = 0,
    private readonly logDir = `./logs/muzero/${config.savedNetworkPath}`
  ) {
    //  Use:
    //    pip install --user tensorboard  # Unless you've already installed it.
    //    C:\Users\mm\AppData\Roaming\Python\Python311\Scripts\tensorboard.exe --logdir ./logs/muzero
    this.tensorBoard = tf.node.tensorBoard(this.logDir, {
      updateFreq: 'epoch',
      histogramFreq: 0
    })
    this.tensorBoard.setParams({
      steps: this.config.trainingSteps,
      batchSize: this.config.batchSize
    })
    this.summary = tf.node.summaryFileWriter(this.logDir)
  }

  public async trainNetwork (storage: SharedStorage, replayBuffer: ReplayBuffer): Promise<void> {
    // Get a private copy of a new untrained network
    const network = storage.initialize()
    // Update the copy with the weights of a potentially loaded network
    storage.latestNetwork().copyWeights(network)
    debug('Training initiated')
    debug(`Training steps: ${this.config.trainingSteps} - starting iteration: ${this.trainingStep}`)
    await this.tensorBoard.onTrainBegin()
    for (let step = 1; step <= this.config.trainingSteps; step++) {
      this.trainingStep++
      if (step % this.config.checkpointInterval === 0) {
        await storage.saveNetwork(step, network)
      }
      await this.tensorBoard.onEpochBegin(this.trainingStep)
      await this.tensorBoard.onBatchBegin(1, {
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
      await this.tensorBoard.onBatchEnd(1, {
        loss,
        accuracy,
        acc_value,
        acc_reward,
        loss_value,
        loss_reward,
        loss_policy
      })
      await this.tensorBoard.onEpochEnd(this.trainingStep, {
        loss,
        accuracy,
        acc_value,
        acc_reward,
        loss_value,
        loss_reward,
        loss_policy
      })
      const performance = replayBuffer.performance()
      // Log the performance measured by number of wins by player 1 in 100 games
      this.summary.scalar('perf', performance, this.trainingStep)
      // Log the current use of tensors. The expected use should include the tensors saved as game history in replay buffer
      this.summary.scalar('use', tf.memory().numTensors, this.trainingStep)
      tf.tidy(() => {
        // Log the predicted policy for the initial state
        // Any game will do, but for simplicity we will use the most recently generated
        const image = replayBuffer.lastGame?.makeImage(0)
        if (image !== undefined) {
          const networkOutput = network.initialInference(new NetworkState(image))
          this.summary.histogram('policy', networkOutput.tfPolicy, this.trainingStep)
        }
      })
      if (info.enabled) {
        info(`--- Performance: ${performance.toFixed(1)}%`)
        info(`--- Accuracy (${this.trainingStep}): ${this.statistics().toFixed(2)}`)
        info(`--- Tensor usage: ${tf.memory().numTensors.toFixed(0)}`)
      }
      await tf.nextFrame()
    }
    await this.tensorBoard.onTrainEnd()
    await storage.saveNetwork(this.config.trainingSteps, network)
  }

  public statistics (): number {
    const mLoss = this.losses.slice(-100)
    const mlossSum = mLoss.reduce((s, l) => s + l, 0)
    return mlossSum / mLoss.length
  }
}
