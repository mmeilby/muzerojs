import type * as tf from '@tensorflow/tfjs-node-gpu'

/**
 * `Action` is an interface made to extend generic `State` objects used in
 * the [[GameRules]] interface. It is meant to ensure that, even though the shape
 * and implementation of the `State` object is left up to the user, it should
 * at least have a `player` property.
 */
export interface Action {
  id: number
  actionShape: number[]
  action: tf.Tensor
  toString: () => string
}
