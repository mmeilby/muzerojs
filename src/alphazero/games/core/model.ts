import { Observation } from '../../networks/nnet'
import { Statewise } from './statewise'

export type GetObservation<State extends Statewise> = (state: State) => Observation

export interface ObservationModel<State extends Statewise> {
  observationSize: number[]
  observation: GetObservation<State>
}
