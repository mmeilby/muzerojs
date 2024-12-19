import { MuZeroNim } from './nim'
import { Muzero } from '../../../muzero'
import * as tf from '@tensorflow/tfjs-node-gpu'

console.log(tf.getBackend())
const factory = new MuZeroNim()
const conf = factory.config()
conf.trainingSteps = 10000
conf.simulations = 10
conf.batchSize = 512
conf.replayBufferSize = 10000
conf.checkpointInterval = 250
conf.lrInit = 0.0005
conf.prioritizedReplay = true
conf.priorityAlpha = 1.0
conf.savedNetworkPath = 'nimv15'
conf.supervisedRL = false
// epochs are only relevant for supervised recurrent learning
// conf.epochs = 2
if (process.env.PRINT !== undefined) {
  new Muzero().printModels(conf)
} else {
  new Muzero().trainingSession(factory, conf)
}
