import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Playerwise } from '../../selfplay/entities'
import { MuZeroNimUtil } from './nimutil'
import { type Action } from '../../selfplay/mctsnode'
import { config } from './nimconfig'

export class MuZeroNimState implements Playerwise {

  constructor (
    public readonly player: number,
    public readonly board: number[],
    public readonly history: Action[]
  ) {
  }

  get observationSize (): number[] {
    return [config.heaps, config.heapSize, 1]
  }

  get observation (): tf.Tensor {
    const board: number[][][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[][] = []
      for (let j = 0; j < config.heapSize; j++) {
        pins[j] = j < this.board[i] ? [1] : [0]
      }
      board.push(pins)
    }
    return tf.tensor3d(board)
  }

  public static action (action: Action): tf.Tensor {
    const support = new MuZeroNimUtil()
    const heap = support.actionToHeap(action.id)
    const nimming = support.actionToNimming(action.id)
    const board: number[][][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[][] = []
      for (let j = 0; j < config.heapSize; j++) {
        pins[j] = heap === i && j < nimming ? [1] : [0]
      }
      board.push(pins)
    }
    return tf.tensor3d(board)
  }

  public static state (observation: tf.Tensor): MuZeroNimState {
    const pins: tf.Tensor = observation.sum(1).reshape([config.heaps])
    return new MuZeroNimState(1, pins.arraySync() as number[], [])
  }

  public toString (): string {
    const support = new MuZeroNimUtil()
    const actionHistory = this.history.length > 0 ? this.history.map(a => a.id).join(':') : '*'
    const actionStringHistory = this.history.length > 0 ? this.history.map(a => support.actionToString(a)).join(',') : '*'
    return `${actionHistory} | ${actionStringHistory} | ${this.board.join('-')}`
  }
}
