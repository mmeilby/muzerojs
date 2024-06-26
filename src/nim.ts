import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { Training } from './muzero/training/training'
import debugFactory from 'debug'
import fs from 'fs'
import { Validate } from './muzero/validation/validate'

const output = debugFactory('muzero:muzero:output')
debugFactory.enable('muzero:muzero:output')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  conf.trainingSteps = 5000
  conf.batchSize = 32
  conf.replayBufferSize = 1024
  conf.checkpointInterval = 100
  conf.rootExplorationFraction = 0.25
  conf.pbCbase = conf.simulations
  conf.lrInit = 0.0005
  output('Running training session for MuZero acting on Nim environment')
  let lastStep = 0
  try {
    const json = fs.readFileSync(`data/${conf.savedNetworkPath}/muzero.json`, { encoding: 'utf8' })
    if (json !== null) {
      lastStep = JSON.parse(json) as number
    }
  } catch (e) {
    // Error: ENOENT: no such file or directory, open 'data/path/muzero.json'
    if ((e as Error).name.includes('ENOENT')) {
      output('Configuration file does not exist. A new file will be created.')
    } else {
      output(e)
    }
  }
  const sharedStorage = new SharedStorage(conf)
  await sharedStorage.loadNetwork()
  const replayBuffer = new ReplayBuffer(conf)
  replayBuffer.loadSavedGames(factory)
  const selfPlay = new SelfPlay(conf, factory)
  const train = new Training(conf, lastStep)
  const validate = new Validate(conf, factory)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer),
    validate.logMeasures(sharedStorage, lastStep)
  ])
  output(`--- Performance: ${replayBuffer.performance().toFixed(1)}%`)
  output(`--- Accuracy (${sharedStorage.networkCount}): ${train.statistics().toFixed(2)}`)
  lastStep += sharedStorage.networkCount
  fs.writeFileSync(`data/${conf.savedNetworkPath}/muzero.json`, JSON.stringify(lastStep), 'utf8')
}

run().then(() => {
}).catch(err => {
  console.error(err)
})
