import { MuZeroModel } from '../core/model'
import { MuZeroNimState } from './nimstate'
import { config } from './nimconfig'

export class NimNetModel implements MuZeroModel<MuZeroNimState> {
  get observationSize (): number {
    return config.heaps * config.heapSize
  }

  public observation (state: MuZeroNimState): number[][] {
    const board: number[][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[] = new Array<number>(config.heapSize).fill(0)
      for (let j = 0; j < state.board[i]; j++) {
        pins[j] = 1
      }
      board.push(pins)
    }
    return board
  }
}
