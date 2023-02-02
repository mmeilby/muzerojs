import { MuZeroSharedStorage } from './muzero/training/sharedstorage'
import { MuZeroAction } from './muzero/games/core/action'
import { MuZeroReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { MuZeroSelfPlay } from './muzero/selfplay/selfplay'
import { MuZeroNim } from './muzero/games/nim/nim'
import { MuZeroNimState } from './muzero/games/nim/nimstate'
import { NimNetModel } from './muzero/games/nim/nimmodel'
import { MuZeroTraining } from './muzero/training/training'
import * as tf from "@tensorflow/tfjs-node";
import debugFactory from 'debug'
import {MuZeroConfig} from "./muzero/games/core/config";

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  conf.lrInit = 0.01
  const replayBuffer = new MuZeroReplayBuffer<MuZeroNimState, MuZeroAction>(conf)
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new MuZeroSharedStorage(conf)
  await sharedStorage.loadNetwork()
  const selfPlay = new MuZeroSelfPlay(conf, factory, model)
  const train = new MuZeroTraining<MuZeroNimState, MuZeroAction>(conf)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
}

run().then(() => {}).catch(err => console.error(err))
