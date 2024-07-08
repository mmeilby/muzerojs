import { MuZeroCartpole } from './cartpoleenv'
import { Muzero } from '../../../muzero'

const factory = new MuZeroCartpole()
const conf = factory.config()
conf.trainingSteps = 10000
conf.batchSize = 32
conf.replayBufferSize = 1024
conf.checkpointInterval = 100
conf.rootExplorationFraction = 0.25
conf.pbCbase = conf.simulations
conf.lrInit = 0.0005
new Muzero().trainingSession(factory, conf)
