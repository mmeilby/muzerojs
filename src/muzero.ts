import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { Training } from './muzero/training/training'
import debugFactory from 'debug'

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  conf.trainingSteps = 256
  conf.batchSize = 16
  conf.replayBufferSize = 16
  conf.checkpointInterval = 25
  const replayBuffer = new ReplayBuffer(conf)
  const sharedStorage = new SharedStorage(conf)
  await sharedStorage.loadNetwork()
  const selfPlay = new SelfPlay(conf, factory)
  const train = new Training(conf)
//  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  //  selfPlay.buildTestHistory(replayBuffer)
  //  replayBuffer.loadSavedGames(factory, model)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer),
    selfPlay.performance(sharedStorage)
  ])
  debug(`--- Performance: ${replayBuffer.statistics().toFixed(1)}%`)
  debug(`--- Accuracy (${sharedStorage.networkCount}): ${train.statistics().toFixed(2)}`)
}

run().then(() => {
}).catch(err => {
  console.error(err)
})
