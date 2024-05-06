import type * as tf from '@tensorflow/tfjs-node-gpu'
import { type Target } from './target'
import { type Action } from '../selfplay/mctsnode'

export class Batch {
  constructor (
    // Observation image for first state in the batch
    public readonly image: tf.Tensor,
    // Sequence of actions played for this batch
    public readonly actions: Action[],
    // Sequence of one-hot vectors corresponding to the actions played for this batch
    public readonly tfActions: tf.Tensor[],
    // Targets for each turn played by executing the corresponding action
    public readonly targets: Target[]
  ) {
  }
}
