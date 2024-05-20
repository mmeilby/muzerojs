import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from '../core/action'

export class MuZeroCartpoleAction implements Action {
  public id: number

  constructor (action?: number) {
    this.id = action ?? -1
  }

  get actionShape (): number[] {
    return [4, 1]
  }

  get action (): tf.Tensor {
    const left = this.id === 0 ? 1 : 0
    const right = this.id === 1 ? 1 : 0
    return tf.tensor2d([[0, left, right, 0]])
  }

  public toString (): string {
    if (this.id < 0) {
      return '?'
    } else {
      return this.id > 0 ? 'Right' : 'Left'
    }
  }
}
