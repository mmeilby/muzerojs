import type * as tf from '@tensorflow/tfjs-node-gpu'
import { type State } from '../games/core/state'

export class NetworkState {
  constructor (
    public readonly hiddenState: tf.Tensor,
    public readonly states?: State[] | undefined
  ) {
  }
}
