import { Actionwise, Playerwise } from '../selfplay/entities'
import { MuZeroGameHistory } from '../selfplay/gamehistory'

export class MuZeroGameSample<State extends Playerwise, Action extends Actionwise> {
  constructor (
    public readonly gameHistory: MuZeroGameHistory<State, Action>,
    public readonly probability: number
  ) {}
}
