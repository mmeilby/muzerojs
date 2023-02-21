import { NimAction } from './nimaction'
import { Statewise } from '../core/statewise'

export class NimState implements Statewise {
  private readonly _key: string
  private readonly _player: number
  private readonly _board: number[]
  private readonly _history: NimAction[]

  constructor (player: number, board: number[], history: NimAction[]) {
    // TODO: Symmetry considerations for state key - should 1|2|3|0|0 be equal to 0|0|1|2|3
    this._key = `${player}#${board.join('|')}/${history.at(-1)?.id ?? '*'}`
    this._player = player
    this._board = board
    this._history = history
  }

  get player (): number {
    return this._player
  }

  get nextPlayer (): number {
    return -this._player
  }

  get board (): number[] {
    return this._board
  }

  get history (): NimAction[] {
    return this._history
  }

  public toString (): string {
    return this._key
  }
}
