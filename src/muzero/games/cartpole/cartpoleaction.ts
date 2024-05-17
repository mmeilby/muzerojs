import type * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from '../core/action'

export class MuZeroCartpoleAction implements Action {
  public id: number

  constructor (action?: number) {
    this.id = action ?? -1
  }

  get actionShape (): number[] {
    return [config.heaps, config.heapSize, 1]
  }

  get action (): tf.Tensor {
  }

  public toString (): string {
    if (this.id < 0) {
      return '?'
    } else {
      return this.id > 0 ? 'Right' : 'Left'
    }
  }
}
