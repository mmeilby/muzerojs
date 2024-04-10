import { type Playerwise } from '../../selfplay/entities'
import { CartPole, type CartPoleDataSet } from './cartpole'
import type { Model } from '../core/model'
import type { Observation } from '../../networks/nnet'
import { NetworkObservation } from '../../networks/network'
import {Action} from "../../selfplay/mctsnode";

export class MuZeroCartpoleState extends CartPole implements Playerwise {
  private readonly _key: string
  private readonly _player: number
  private readonly _dataset: CartPoleDataSet
  private readonly _history: Action[]

  constructor (dataset: CartPoleDataSet, history: Action[]) {
    super()
    this._key = history.length > 0 ? history.map(a => a.id).join(':') : '*'
    this._player = 1
    this._dataset = dataset
    this._history = history
  }

  get player (): number {
    return this._player
  }

  get dataset (): CartPoleDataSet {
    return this._dataset
  }

  get history (): Action[] {
    return this._history
  }

  public toString (): string {
    return `${this._key} | ${this._history.length > 0 ? this._history.map(a => a.id > 0 ? 'right' : 'left').join(':') : '*'} | ${super.toString(this._dataset)}`
  }
}

export class CartpoleNetModel implements Model<MuZeroCartpoleState> {
  public readonly observationSize = 4

  public observation (state: MuZeroCartpoleState): Observation {
    return new NetworkObservation([state.dataset.x, state.dataset.xDot, state.dataset.theta, state.dataset.thetaDot])
  }
}
