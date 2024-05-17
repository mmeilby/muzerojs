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
    const heap = this.actionToHeap()
    const nimming = this.actionToNimming()
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

  public toString (): string {
    if (this.id < 0) {
      return 'H?-?'
    }
    const heap = this.actionToHeap()
    const nimmingSize = this.actionToNimming()
    return `H${heap + 1}-${nimmingSize + 1}`
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

  public actionToHeap (): number {
    return this.reduce(this.id)
  }

  public actionToNimming (): number {
    return this.nimming(this.id)
  }

  private reduce (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.reduce(n - util.heapMap[level], level + 1) : level
  }

  private nimming (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.nimming(n - util.heapMap[level], level + 1) : n
  }
}
