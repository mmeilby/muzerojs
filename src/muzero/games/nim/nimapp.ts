import { MuZeroNim } from './nim'
import { Muzero } from '../../../muzero'

const factory = new MuZeroNim()
const conf = factory.config()
conf.trainingSteps = 20000
conf.batchSize = 16
conf.epochs = 3
conf.replayBufferSize = 1024
conf.checkpointInterval = 100
conf.lrInit = 0.005
conf.savedNetworkPath = 'nimv5'
conf.supervisedRL = true
if (process.env.PRINT !== undefined) {
  new Muzero().printModels(conf)
} else {
  new Muzero().trainingSession(factory, conf)
}
