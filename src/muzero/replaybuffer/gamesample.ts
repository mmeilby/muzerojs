import { GameHistory } from '../selfplay/gamehistory'

export class MuZeroGameSample {
  constructor (
    public readonly gameHistory: GameHistory,
    public readonly probability: number
  ) {
  }
}
