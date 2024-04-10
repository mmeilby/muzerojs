import { type HiddenState } from '../networks/nnet'

/**
 * `Playerwise` is an interface made to extend generic `State` objects used in
 * the [[GameRules]] interface. It is meant to ensure that, even though the shape
 * and implementation of the `State` object is left up to the user, it should
 * at least have a `player` property.
 */
export interface Playerwise {
  player: number
  toString: () => string
}

export interface Actionwise {
  id: number
}

export class Normalizer {
  constructor (
    private min_ = Infinity,
    private max_ = -Infinity
  ) {
  }

  update (value: number): void {
    this.min_ = Math.min(this.min_, value)
    this.max_ = Math.max(this.max_, value)
  }

  normalize (value: number): number {
    return this.max_ > this.min_
      ? (value - this.min_) / (this.max_ - this.min_)
      : value
  }
}
