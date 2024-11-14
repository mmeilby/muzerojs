import { MuZeroNim } from './nim'
import { Muzero } from '../../../muzero'

const factory = new MuZeroNim()
const conf = factory.config()
conf.trainingSteps = 5000
conf.simulations = 10
conf.batchSize = 16
conf.epochs = 2
conf.replayBufferSize = 1024
conf.checkpointInterval = 100
conf.lrInit = 0.0005
conf.savedNetworkPath = 'nimv15'
conf.supervisedRL = false
if (process.env.PRINT !== undefined) {
  new Muzero().printModels(conf)
} else {
  new Muzero().trainingSession(factory, conf)
}
