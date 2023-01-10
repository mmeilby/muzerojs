import { MuZeroAction } from './action'
import { ApplyAction, CalculateReward, GenerateActions, Playerwise, StateIsTerminal } from '../../selfplay/entities'

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
export interface MuZeroEnvironment<State extends Playerwise, Action> {
  config: () => {
    actionSpaceSize: number
    boardSize: number
  }
  reset: () => State
  step: ApplyAction<State, Action>
  legalActions: GenerateActions<State, Action>
  terminal: StateIsTerminal<State>
  reward: CalculateReward<State>
  expertAction: (state: State) => MuZeroAction
  deserialize: (stream: string) => State
  serialize: (state: State) => string
  toString: (state: State) => string
}
