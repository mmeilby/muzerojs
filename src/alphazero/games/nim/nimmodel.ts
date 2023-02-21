import { ObservationModel } from '../core/model'
import { NimState } from './nimstate'
import { config } from './nimconfig'
import { Observation } from '../../networks/nnet'
import { MuZeroNetObservation } from '../../networks/network'

export class NimNetModel implements ObservationModel<NimState> {
  get observationSize (): number[] {
    return [config.heaps * 3, 3]
  }

  public observation (state: NimState): Observation {
    const dicePatterns: number[][][] = [
      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
      [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
      [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
      [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
      [[1, 0, 1], [1, 1, 1], [1, 0, 1]],
      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
      [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ]
    const board: number[][][] = [[], [], []]
    for (let i = 0; i < config.heaps; i++) {
      for (let j = 0; j < 3; j++) {
        board[j].push(dicePatterns[state.board[i]][j])
      }
    }
    return new MuZeroNetObservation(board.flatMap(dice => dice))
  }
}
