import { util } from './nimconfig'

import { type Action } from '../core/action'
import { MuZeroNimAction } from './nimaction'

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class MuZeroNimUtil {
  public actionToHeap (action: number): number {
    return this.reduce(action)
  }

  public actionToNimming (action: number): number {
    return this.nimming(action)
  }

  public heapNimmingToAction (heap: number, nimming: number): Action {
    let action = nimming
    for (let h = 0; h < heap; h++) {
      action += util.heapMap[h]
    }
    return new MuZeroNimAction(action)
  }

  public actionToString (action: Action): string {
    if (action.id < 0) {
      return 'H?-?'
    }
    const heap = this.actionToHeap(action.id)
    const nimmingSize = this.actionToNimming(action.id)
    return `H${heap + 1}-${nimmingSize + 1}`
  }

  public actionFromString (action: string): Action {
    const [sHeap, sNimming] = action.split('-')
    if (sHeap.includes('?') && sNimming.includes('?')) {
      return new MuZeroNimAction()
    } else {
      return this.heapNimmingToAction(Number.parseInt(sHeap.slice(1)) - 1, Number.parseInt(sNimming) - 1)
    }
  }

  private reduce (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.reduce(n - util.heapMap[level], level + 1) : level
  }

  private nimming (n: number, level = 0): number {
    return n >= util.heapMap[level] ? this.nimming(n - util.heapMap[level], level + 1) : n
  }
}
