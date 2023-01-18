import { Playerwise } from '../../selfplay/entities'

export type GetObservation<State extends Playerwise> = (state: State) => number[][]

export interface MuZeroModel<State extends Playerwise> {
  observationSize: number
  observation: GetObservation<State>
}
