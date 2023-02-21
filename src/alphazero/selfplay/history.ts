import { Statewise } from '../games/core/statewise'
import { Actionwise } from '../games/core/actionwise'

export class HistoryObject<State extends Statewise, Action extends Actionwise> {
  // Policy for the state
  private pi_: number[]
  // Reward achieved for this game - polarized for the current player
  private value_: number

  /**
     * Create a game history object
     * @param state_ The state that was the origin of this game move
     * @param action_ The action that was chosen as the best from the policy
     */
  constructor (
    private readonly state_: State,
    private readonly action_: Action
  ) {
    this.pi_ = []
    this.value_ = 0
  }

  get state (): State {
    return this.state_
  }

  get action (): Action {
    return this.action_
  }

  get pi (): number[] {
    return this.pi_
  }

  set pi (policy: number[]) {
    this.pi_ = policy
  }

  get value (): number {
    return this.value_
  }

  set value (reward: number) {
    this.value_ = reward
  }

  get player (): number {
    return this.state_.player
  }
}
