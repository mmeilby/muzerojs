import * as tf from '@tensorflow/tfjs-node'
import { Playerwise } from '../../selfplay/entities'

export type GetObservation<State extends Playerwise> = (state: State) => tf.Tensor

export interface MuZeroModel<State extends Playerwise> {
  observationSize: number
  observation: GetObservation<State>
}
