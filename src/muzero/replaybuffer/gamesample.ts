import { Playerwise } from '../selfplay/entities'
import { GameHistory } from '../selfplay/gamehistory'

export class MuZeroGameSample<State extends Playerwise> {
  constructor (
    public readonly gameHistory: GameHistory<State>,
    public readonly probability: number
  ) {}
}
