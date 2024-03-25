import { type MuZeroModel } from '../core/model'
import { type MuZeroNimState } from './nimstate'
import { config, util } from './nimconfig'
import { type MuZeroObservation } from '../../networks/nnet'
import { MuZeroNetObservation } from '../../networks/network'

export class NimNetModel implements MuZeroModel<MuZeroNimState> {
  get observationSize (): number {
    return util.heapMap.reduce((s, h) => s + h, 0)
  }

  public observation (state: MuZeroNimState): MuZeroObservation {
    const board: number[][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[] = new Array<number>(util.heapMap[i]).fill(0)
      for (let j = 0; j < state.board[i]; j++) {
        pins[j] = 1
      }
      board.push(pins)
    }
    return new MuZeroNetObservation(board.flat())
  }
}
