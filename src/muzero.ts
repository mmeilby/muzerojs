import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { type MuZeroNimState } from './muzero/games/nim/nimstate'
import { Training } from './muzero/training/training'
import * as tf from '@tensorflow/tfjs-node'
import debugFactory from 'debug'

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  conf.trainingSteps = 200000
  conf.batchSize = 64
  conf.replayBufferSize = 128
  conf.checkpointInterval = 25
  const replayBuffer = new ReplayBuffer<MuZeroNimState>(conf)
  const sharedStorage = new SharedStorage(conf)
  //  await sharedStorage.loadNetwork()
  const selfPlay = new SelfPlay<MuZeroNimState>(conf, factory)
  const train = new Training<MuZeroNimState>(conf)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  //  selfPlay.buildTestHistory(replayBuffer)
  //  replayBuffer.loadSavedGames(factory, model)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
  debug(`--- Performance: ${replayBuffer.statistics().toFixed(1)}%`)
  debug(`--- Accuracy (${sharedStorage.networkCount}): ${train.statistics().toFixed(2)}`)
}

run().then(() => {
}).catch(err => {
  console.error(err)
})
