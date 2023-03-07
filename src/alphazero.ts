import { Nim } from './alphazero/games/nim/nim'
import { NimNetModel } from './alphazero/games/nim/nimmodel'
import { Config } from './alphazero/games/core/config'
import { Coach } from './alphazero/coach/coach'
import { NNet } from './alphazero/networks/network'
import * as tf from '@tensorflow/tfjs-node-gpu'

import debugFactory from 'debug'
const debug = debugFactory('alphazero:alphazero:debug')

async function run (): Promise<void> {
  const factory = new Nim()
  const model = new NimNetModel() // NimNetMockedModel()
  const config = factory.config()
  const conf = new Config(config.actionSpaceSize, model.observationSize)
  conf.replayBufferSize = 100
  conf.numUnrollSteps = 1
  conf.numEpisodes = 100
  conf.batchSize = 200
  conf.validationSize = 40
  conf.gradientUpdateFreq = 10
  conf.lrInit = 0.001
  conf.epochs = 100
  conf.temperatureThreshold = 15
  conf.networkUpdateThreshold = 0.5
  conf.numIterations = 1000
  conf.simulations = 25
  conf.numGames = 40
  conf.trainingSteps = 1
  /*
  const replayBuffer = new ReplayBuffer<NimState, NimAction>(conf)
  replayBuffer.loadSavedGames(factory, model)
  const sharedStorage = new SharedStorage(conf)
  await sharedStorage.loadNetwork()
  */
  const coach = new Coach(conf, factory, model)
  const nnet = new NNet(conf) // MockedNetwork(factory)
  debug(`Tensor usage baseline: ${tf.memory().numTensors}`)
  await coach.learn(nnet)
}

run().then(() => {}).catch(err => console.error(err))
