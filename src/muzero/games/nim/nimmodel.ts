import { MuZeroModel } from '../core/model'
import { MuZeroNimState } from './nimstate'
import { config } from './nimconfig'
import {MuZeroObservation} from "../../networks/nnet";
import {MuZeroNetObservation} from "../../networks/network";

export class NimNetModel implements MuZeroModel<MuZeroNimState> {
  get observationSize (): number {
    return config.heaps * config.heapSize
  }

  public observation (state: MuZeroNimState): MuZeroObservation {
    const board: number[][] = []
    for (let i = 0; i < config.heaps; i++) {
      const pins: number[] = new Array<number>(config.heapSize).fill(0)
      for (let j = 0; j < state.board[i]; j++) {
        pins[j] = 1
      }
      board.push(pins)
    }
    return new MuZeroNetObservation(board)
  }
}
