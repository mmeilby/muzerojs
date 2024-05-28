import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { Training } from './muzero/training/training'
import debugFactory from 'debug'
import fs from 'fs'

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  conf.trainingSteps = 100
  conf.batchSize = 32
  conf.replayBufferSize = 256
  conf.checkpointInterval = 25
  conf.rootExplorationFraction = 0.25
  conf.pbCbase = conf.simulations
  conf.lrInit = 0.0005
  let lastStep = 0
  try {
    const json = fs.readFileSync(`data/${conf.savedNetworkPath}/muzero.json`, { encoding: 'utf8' })
    if (json !== null) {
      lastStep = JSON.parse(json) as number
    }
  } catch (e) {
    debug(e)
  }
  const sharedStorage = new SharedStorage(conf)
  await sharedStorage.loadNetwork()
  const replayBuffer = new ReplayBuffer(conf)
  replayBuffer.loadSavedGames(factory)
  const selfPlay = new SelfPlay(conf, factory)
  const train = new Training(conf, lastStep)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer),
    selfPlay.performance(sharedStorage)
  ])
  debug(`--- Performance: ${replayBuffer.statistics().toFixed(1)}%`)
  debug(`--- Accuracy (${sharedStorage.networkCount}): ${train.statistics().toFixed(2)}`)
  lastStep += sharedStorage.networkCount
  fs.writeFileSync(`data/${conf.savedNetworkPath}/muzero.json`, JSON.stringify(lastStep), 'utf8')
}

run().then(() => {
}).catch(err => {
  console.error(err)
})
