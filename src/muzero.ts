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

const debug = debugFactory('muzero:muzero:debug')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const replayBuffer = new MuZeroReplayBuffer<MuZeroNimState, MuZeroAction>({
    replayBufferSize: 200,
    actionSpace: config.actionSpaceSize,
    tdSteps: config.actionSpaceSize
  })
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new MuZeroSharedStorage({
    observationSize: model.observationSize,
    actionSpaceSize: config.actionSpaceSize
  })
  await sharedStorage.loadNetwork()
  const selfPlay = new MuZeroSelfPlay({
    selfPlaySteps: 1000,
    actionSpaceSize: config.actionSpaceSize,
    maxMoves: config.actionSpaceSize,
    simulations: 100
  }, factory, model)
  const train = new MuZeroTraining<MuZeroNimState, MuZeroAction>({
    trainingSteps: 5000,
    checkpointInterval: 25,
    tdSteps: config.actionSpaceSize,
    learningRate: 0.001
  })
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  await Promise.all([
    selfPlay.runSelfPlay(sharedStorage, replayBuffer),
    train.trainNetwork(sharedStorage, replayBuffer)
  ])
}

run().then(() => {}).catch(err => console.error(err))
