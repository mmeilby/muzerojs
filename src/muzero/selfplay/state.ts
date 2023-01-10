import { Playerwise } from './entities'

export class MuZeroState implements Playerwise {
  private readonly _player: number
  private readonly _state: string
  private readonly _history: number[]
  private readonly _key: string

  constructor (player: number, state: string, history: number[]) {
    this._player = player
    this._state = state
    this._history = history
    this._key = history.join(':')
  }

  get player (): number {
    return this._player
  }

  get state (): string {
    return this._state
  }

  get history (): number[] {
    return this._history
  }

  toString (): string {
    return this._key
  }
}
