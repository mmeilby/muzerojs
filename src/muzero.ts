import { MuZeroSharedStorage } from './muzero/training/sharedstorage'
import { type MuZeroAction } from './muzero/games/core/action'
import { MuZeroReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { MuZeroSelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { type MuZeroNimState } from './muzero/games/nim/nimstate'
import { NimNetModel } from './muzero/games/nim/nimmodel'
import { MuZeroTraining } from './muzero/training/training'
import * as tf from '@tensorflow/tfjs-node'
import debugFactory from 'debug'
import { MuZeroConfig } from './muzero/games/core/config'
import { MuZeroCartpole } from './muzero/games/cartpole/cartpoleenv'
import { CartpoleNetModel, type MuZeroCartpoleState } from './muzero/games/cartpole/cartpolestate'

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  conf.maxMoves = 500
  conf.decayingParam = 0.997
  conf.rootDirichletAlpha = 0.25
  conf.simulations = 150
  conf.batchSize = 100
  conf.tdSteps = 7
  conf.lrInit = 0.0001
  conf.trainingSteps = 200
  conf.replayBufferSize = 1000
  conf.numUnrollSteps = 500
  conf.lrDecayRate = 0.1
  const replayBuffer = new MuZeroReplayBuffer<MuZeroNimState, MuZeroAction>(conf)
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new MuZeroSharedStorage<MuZeroAction>(conf)
  await sharedStorage.loadNetwork()
  const selfPlay = new MuZeroSelfPlay<MuZeroNimState, MuZeroAction>(conf, factory, model)
  const train = new MuZeroTraining<MuZeroNimState, MuZeroAction>(conf)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  for (let sim = 0; sim < conf.selfPlaySteps; sim++) {
    const useBaseline = tf.memory().numTensors
    const network = sharedStorage.latestNetwork()
    await selfPlay.selfPlay(network, replayBuffer)
    await train.trainSingleInteration(sharedStorage, replayBuffer)
    if (tf.memory().numTensors - useBaseline > 0) {
      debug(`TENSOR USAGE IS GROWING: ${tf.memory().numTensors - useBaseline} (total: ${tf.memory().numTensors})`)
    }
  }
  /*
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
   */
}

run().then(() => {}).catch(err => { console.error(err) })
