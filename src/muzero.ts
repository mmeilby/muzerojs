import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { type MuZeroNimState } from './muzero/games/nim/nimstate'
import { NimNetModel } from './muzero/games/nim/nimmodel'
import { Training } from './muzero/training/training'
import * as tf from '@tensorflow/tfjs-node'
import debugFactory from 'debug'

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const conf = factory.config()
  conf.selfPlaySteps = 200
  conf.trainingSteps = 200
  conf.batchSize = 32
  conf.replayBufferSize = 32
  const replayBuffer = new ReplayBuffer<MuZeroNimState>(conf)
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new SharedStorage(conf)
  await sharedStorage.loadNetwork()
  const selfPlay = new SelfPlay<MuZeroNimState>(conf, factory, model)
  const train = new Training<MuZeroNimState>(conf)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
/*
  for (let sim = 0; sim < conf.selfPlaySteps; sim++) {
    const useBaseline = tf.memory().numTensors
    const network = sharedStorage.latestNetwork()
    await selfPlay.selfPlay(network, replayBuffer)
    await train.trainSingleInteration(sharedStorage, replayBuffer)
    if (tf.memory().numTensors - useBaseline > 0) {
      debug(`TENSOR USAGE IS GROWING: ${tf.memory().numTensors - useBaseline} (total: ${tf.memory().numTensors})`)
    }
 */
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
}

run().then(() => {}).catch(err => { console.error(err) })
