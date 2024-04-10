import { type Model } from '../core/model'
import { type MuZeroNimState } from './nimstate'
import { config, util } from './nimconfig'
import { type Observation } from '../../networks/nnet'
import { NetworkObservation } from '../../networks/network'

export class NimNetModel implements Model<MuZeroNimState> {
  get observationSize (): number {
    return util.heapMap.reduce((s, h) => s + h, 0)
  }

  public observation (state: MuZeroNimState): Observation {
    const board: number[][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[] = new Array<number>(util.heapMap[i]).fill(0)
      for (let j = 0; j < state.board[i]; j++) {
        pins[j] = 1
      }
      board.push(pins)
    }
    return new NetworkObservation(board.flat())
  }
}
