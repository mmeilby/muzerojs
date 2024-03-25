import { type Playerwise } from '../../selfplay/entities'
import { type MuZeroObservation } from '../../networks/nnet'

export type GetObservation<State extends Playerwise> = (state: State) => MuZeroObservation

export interface MuZeroModel<State extends Playerwise> {
  observationSize: number
  observation: GetObservation<State>
}
