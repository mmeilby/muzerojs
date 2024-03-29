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
  const factory = new MuZeroCartpole()
  const model = new CartpoleNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  conf.lrInit = 0.0001
  conf.decayingParam = 0.997
  conf.selfPlaySteps = 150
  conf.replayBufferSize = 100
  conf.maxMoves = 100
  const replayBuffer = new MuZeroReplayBuffer<MuZeroCartpoleState, MuZeroAction>(conf)
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new MuZeroSharedStorage<MuZeroAction>(conf)
  await sharedStorage.loadNetwork()
  const selfPlay = new MuZeroSelfPlay<MuZeroCartpoleState, MuZeroAction>(conf, factory, model)
  const train = new MuZeroTraining<MuZeroCartpoleState, MuZeroAction>(conf)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  for (let sim = 0; sim < conf.selfPlaySteps; sim++) {
    const network = sharedStorage.latestNetwork()
    await selfPlay.selfPlay(network, replayBuffer)
    await train.trainSingleInteration(sharedStorage, replayBuffer)
  }
  /*
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
   */
}

run().then(() => {}).catch(err => { console.error(err) })
