import { Playerwise } from '../../selfplay/entities'
import { MuZeroAction } from '../core/action'
import { config } from './nimconfig'

export class MuZeroNimState implements Playerwise {
  private readonly _key: string
  private readonly _player: number
  private readonly _board: number[]
  private readonly _history: MuZeroAction[]

  constructor (player: number, board: number[], history: MuZeroAction[]) {
    this._key = history.length > 0 ? history.map(a => a.id).join(':') : '*'
    this._player = player
    this._board = board
    this._history = history
  }

  get player (): number {
    return this._player
  }

  get board (): number[] {
    return this._board
  }

  get history (): MuZeroAction[] {
    return this._history
  }

  public actionToString (id: number): string {
    if (id < 0) {
      return 'H?-?'
    }
    const heap = Math.floor(id / config.heapSize) + 1
    const nimmingSize = id % config.heapSize + 1
    return `H${heap}-${nimmingSize}`
  }

  public toString (): string {
    return `${this._key} | ${this._history.length > 0 ? this._history.map(a => this.actionToString(a.id)).join(':') : '*'} | ${this._board.join('-')}`
  }
}
