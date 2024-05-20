import type * as tf from '@tensorflow/tfjs-node-gpu'
import { State } from '../games/core/state'
import { Action } from '../games/core/action'

export class TensorNetworkOutput {
  public state: State[] | undefined

  constructor (
    public tfValue: tf.Tensor,
    public tfReward: tf.Tensor,
    public tfPolicy: tf.Tensor,
    public tfHiddenState: tf.Tensor
  ) {
  }
}

export class NetworkInput {
  public state: State[] | undefined
  public action: Action[] | undefined

  constructor (
    public tfState: tf.Tensor,
    public tfAction?: tf.Tensor | undefined
  ) {
  }
}
