import type * as tf from '@tensorflow/tfjs-node'
import { type Playerwise } from '../../selfplay/entities'

export interface Model<State extends Playerwise> {
  observationSize: number
  observation: (state: State) => tf.Tensor
}
