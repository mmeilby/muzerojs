import * as tf from '@tensorflow/tfjs-node-gpu'
import { config, util } from './nimconfig'
import { type Action } from '../core/action'

export class MuZeroNimAction implements Action {
  public id: number

  constructor (action?: number) {
    this.id = action ?? -1
  }

  get actionShape (): number[] {
    return [config.heaps, config.heapSize, 1]
  }

  get action (): tf.Tensor {
    const heap = this.heap
    const nimming = this.nimming
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

  public get heap (): number {
    return this.rHeap(this.id)
  }

  public get nimming (): number {
    return this.rNimming(this.id) + 1
  }

  public toString (): string {
    if (this.id < 0) {
      return 'H?-?'
    }
    return `H${this.heap + 1}-${this.nimming}`
  }

  public set (action: string): MuZeroNimAction {
    const [sHeap, sNimming] = action.split('-')
    if (sHeap.includes('?') && sNimming.includes('?')) {
      this.id = -1
    } else {
      this.id = Number.parseInt(sNimming) - 1
      const heap = Number.parseInt(sHeap.slice(1)) - 1
      for (let h = 0; h < heap; h++) {
        this.id += util.heapMap[h]
      }
      if (Number.isNaN(this.id)) {
        this.id = -1
      }
    }
    return this
  }

  public preset (heap: number, nimming: number): MuZeroNimAction {
    let action = nimming
    for (let h = 0; h < heap; h++) {
      action += util.heapMap[h]
    }
    this.id = action
    return this
  }

  private rHeap (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.rHeap(n - util.heapMap[level], level + 1) : level
  }

  private rNimming (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.rNimming(n - util.heapMap[level], level + 1) : n
  }
}
