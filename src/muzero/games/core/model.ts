import { type Playerwise } from '../../selfplay/entities'
import { type Observation } from '../../networks/nnet'

export interface Model<State extends Playerwise> {
  observationSize: number
  observation (state: State): Observation
}
