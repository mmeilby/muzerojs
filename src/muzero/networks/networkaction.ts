import type * as tf from '@tensorflow/tfjs-node-gpu'
import { Action } from '../games/core/action'

export class NetworkAction {
  constructor (
    public readonly tfAction: tf.Tensor,
    public readonly actions?: Action[] | undefined
  ) {
  }
}
