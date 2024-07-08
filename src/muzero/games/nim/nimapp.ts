import { MuZeroNim } from './nim'
import { Muzero } from '../../../muzero'

const factory = new MuZeroNim()
const conf = factory.config()
conf.trainingSteps = 10000
conf.batchSize = 32
conf.replayBufferSize = 1024
conf.checkpointInterval = 100
conf.lrInit = 0.0005
conf.savedNetworkPath = 'nimv2'
new Muzero().trainingSession(factory, conf)
