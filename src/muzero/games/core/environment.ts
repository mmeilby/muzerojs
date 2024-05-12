import * as tf from '@tensorflow/tfjs-node-gpu'
import { Playerwise } from '../../selfplay/entities'
import { Action } from '../../selfplay/mctsnode'
import { Config } from './config'

/**
 * MuZeroEnvironment - The environment MuZero is interacting with
 *
 * The environment defines a set of methods to interact with the implemented system/game
 *   - configuration parameters
 *   - reset of the system/game
 *   - applying an action to the system/game
 *   - reward and end-state of the last action applied
 *   - data management of the trained network
 *   - best guess for best next action
 *   - logging
 */
export interface Environment<State extends Playerwise> {
  config (): Config

  reset (): State

  /*
  `step` is a type of function that you provide that takes in a `State`
  and an `Action` as arguments. It applies the `Action` to the `State` and returns
  a new `State`.

  **IMPORTANT**
  Make sure that the function indeed returns a NEW State and does not simply
  mutate the provided State.
   */
  step (state: State, action: Action): State

  /*
  `legalActions` is a type of function that you provide that takes in a `State`
  as an argument and returns an `Array` of possible `Action`s.
  */
  legalActions (state: State): Action[]

  /*
  `terminal` is a type of function that you provide that takes in a `State`
  as an argument and returns `true` if the game is over and `false` otherwise.
   */
  terminal (state: State): boolean

  /*
  `reward` is a type of function that takes in a `State`
  and a `number` representing the player, as arguments. Given the game `State`,
  it calculates a reward for the player and returns that reward as a `number`.

  Normally, you would want a win to return 1, a loss to return -1 and a draw
  to return 0, but you can decide on a different reward scheme.
   */
  reward (state: State, player: number): number

  expertAction (state: State): Action

  expertActionPolicy (state: State): tf.Tensor

  deserialize (stream: string): State

  serialize (state: State): string

  toString (state: State): string
}
