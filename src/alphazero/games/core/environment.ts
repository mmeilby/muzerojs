import { Statewise } from './statewise'
import { Actionwise } from './actionwise'

/**
 * Environment - The environment the network is interacting with
 *
 * The environment defines a set of methods to interact with the implemented system/game
 *   - configuration parameters
 *   - reset of the system/game
 *   - applying an action to the system/game
 *   - reward and end-state of the last action applied
 *   - data.old.copy(1) management of the trained network
 *   - best guess for best next action
 *   - logging
 */
export interface Environment<State extends Statewise, Action extends Actionwise> {
  config: () => {
    actionSpaceSize: number
    boardSize: number
  }
  reset: () => State
  /**
   * `ApplyAction` is a type of function that you provide that takes in a `State`
   * and an `Action` as arguments. It applies the `Action` to the `State` and returns
   * a new `State`.
   *
   * **IMPORTANT**
   * Make sure that the function indeed returns a NEW State and does not simply
   * mutate the provided State.
   *
   * ### Example
   * ```javascript
   * function(state, action) {
   *   let newState;
   *
   *   // Apply the action to state and create a new State object.
   *
   *   return newState;
   * }
   * ```
   * @param state An object representing the state of the game.
   * @param action An object representing an action in the game.
   */
  step: (state: State, action: Action) => State
  /**
   * `GenerateActions` is a type of function that you provide that takes in a `State`
   * as an argument and returns an `Array` of possible `Action`s.
   *
   * ### Example
   * ```javascript
   * function(state) {
   *   const possibleActions = [];
   *
   *   // Some kind of algortihm that you implement and
   *   // pushes all possible Action(s) into an array.
   *
   *   return possibleActions;
   * }
   * ```
   * @param state An object representing the state of the game.
   */
  legalActions: (state: State) => Action[]
  /**
   * `StateIsTerminal` is a type of function that you provide that takes in a `State`
   * as an argument and returns `true` if the game is over and `false` otherwise.
   *
   * ### Example
   * ```javascript
   * function(state) {
   *   if (gameIsADraw(state) || gamesIsWon(state)) return true;
   *
   *   return false;
   * }
   * ```
   * @param state An object representing the state of the game.
   */
  terminal: (state: State) => boolean
  /**
   * `CalculateReward` is a type of function that you provide that takes in a `State`
   * and a `number` representing the player, as arguments. Given the game `State`,
   * it calculates a reward for the player and returns that reward as a `number`.
   *
   * Normaly, you would want a win to return 1, a loss to return -1 and a draw
   * to return 0 but you can decide on a different reward scheme.
   *
   * ### Example
   * ```javascript
   * function(state, player) {
   *   if (hasWon(state, player)) return 1;
   *
   *   if (isDraw(state)) return 0;
   *
   *   return -1;
   * }
   * ```
   * @param state An object representing the state of the game.
   * @param player
   */
  reward: (state: State, player: number) => number
  expertAction: (state: State) => number[]
  action: (id: number) => Action
  actionToString: (id: number) => string
  deserialize: (stream: string) => State
  serialize: (state: State) => string
  toString: (state: State) => string
}
