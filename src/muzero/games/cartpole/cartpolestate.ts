import { CartPole, type CartPoleDataSet } from './cartpole'
import * as tf from '@tensorflow/tfjs-node-gpu'
import { type State } from '../core/state'
import { type Action } from '../core/action'

export class MuZeroCartpoleState extends CartPole implements State {
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

  get observationShape (): number[] {
    return [4, 1]
  }

  get observation (): tf.Tensor {
    return tf.tensor2d([[this._dataset.x, this._dataset.xDot, this._dataset.theta, this._dataset.thetaDot]])
  }

  public toString (): string {
    return `${this._key} | ${this._history.length > 0 ? this._history.map(a => a.id > 0 ? 'right' : 'left').join(':') : '*'} | ${super.toString(this._dataset)}`
  }
}
