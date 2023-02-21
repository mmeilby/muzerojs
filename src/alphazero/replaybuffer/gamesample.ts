import { GameHistory } from '../selfplay/gamehistory'
import { Actionwise } from '../games/core/actionwise'
import { Statewise } from '../games/core/statewise'

export class GameSample<State extends Statewise, Action extends Actionwise> {
  constructor (
    public readonly gameHistory: GameHistory<State, Action>,
    public readonly probability: number
  ) {}
}
