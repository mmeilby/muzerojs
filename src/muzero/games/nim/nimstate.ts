import * as tf from '@tensorflow/tfjs-node-gpu'
import { config } from './nimconfig'
import { type State } from '../core/state'
import { type Action } from '../core/action'

export class MuZeroNimState implements State {
  constructor (
    public readonly player: number,
    public readonly board: number[],
    public readonly history: Action[]
  ) {
  }

  /**
   * Return the shape of the observation tensors excluding the batch dimension
   */
  get observationShape (): number[] {
    return [config.heaps, config.heapSize, 1]
  }

  /**
   * Make an observation tensor shaped as observationShape including the batch dimension
   */
  get observation (): tf.Tensor {
    const board: number[][][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[][] = []
      for (let j = 0; j < config.heapSize; j++) {
        pins[j] = j < this.board[i] ? [1] : [0]
      }
      board.push(pins)
    }
    return tf.tidy(() => tf.tensor3d(board).expandDims(0))
  }

  public static state (observation: tf.Tensor): MuZeroNimState {
    const pins: tf.Tensor = observation.sum(1).reshape([config.heaps])
    return new MuZeroNimState(1, pins.arraySync() as number[], [])
  }

  public toString (): string {
    const actionHistory = this.history.length > 0 ? this.history.map(a => a.id).join(':') : '*'
    const actionStringHistory = this.history.length > 0 ? this.history.map(a => a.toString()).join(',') : '*'
    return `${actionHistory} | ${actionStringHistory} | ${this.board.join('-')}`
  }
}
