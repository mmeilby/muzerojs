import { MuZeroEnvironment } from '../games/core/environment'
import { Actionwise, MCTSNode, Playerwise } from './entities'
import { MuZeroModel } from '../games/core/model'
import { MuZeroTarget } from '../replaybuffer/target'

interface MuZeroGameHistoryObject {
  actionHistory: number[]
  childVisits: number[][]
  rootValues: number[]
  priorities: number[]
  gamePriority: number
}
export class MuZeroGameHistory<State extends Playerwise, Action extends Actionwise> {
  private readonly environment: MuZeroEnvironment<State, Action> // Game specific environment.
  private readonly model: MuZeroModel<State>
  private _state: State
  // A list of observation input tensors for the representation network at each turn of the game
  private readonly observationHistory: number[][][]
  // A list of actions taken at each turn of the game
  public readonly actionHistory: Action[]
  // A list of true rewards received at each turn of the game
  public readonly rewards: number[]
  // A list of tracked player at each turn of the game
  public readonly toPlayHistory: number[]
  // A list of action probability distributions from the root node at each turn of the game
  // Derived from number of visits per root children per action
  public readonly childVisits: number[][]
  // A list of values of the root node at each turn of the game
  public readonly rootValues: number[]
  // Root value deviations from true target value - used for priority sorting of saved games
  public priorities: number[]
  // Largest absolute deviation from target value of the priorities list
  public gamePriority: number

  constructor (
    environment: MuZeroEnvironment<State, Action>,
    model: MuZeroModel<State>
  ) {
    this.environment = environment
    this.model = model
    this._state = environment.reset()
    this.observationHistory = []
    this.actionHistory = []
    this.rewards = []
    this.toPlayHistory = []
    this.childVisits = []
    this.rootValues = []
    this.priorities = []
    this.gamePriority = 0
  }

  get state (): State {
    return this._state
  }

  public terminal (): boolean {
    // Game specific termination rules.
    return this.environment.terminal(this._state)
  }

  public legalActions (): Action[] {
    // Game specific calculation of legal actions.
    return this.environment.legalActions(this._state)
  }

  public apply (action: Action): State {
    this.observationHistory.push(this.model.observation(this._state))
    const state = this.environment.step(this._state, action)
    this.rewards.push(this.environment.reward(state, state.player))
    this.actionHistory.push(action)
    this.toPlayHistory.push(state.player)
    this._state = state
    return state
  }

  public storeSearchStatistics (rootNode: MCTSNode<State, Action>): void {
    this.childVisits.push(rootNode.policy(this.environment.config().actionSpaceSize))
    this.rootValues.push(rootNode.mctsState.value)
  }

  public makeImage (stateIndex: number): number[][] {
    // Game specific feature planes.
    // Convert to positive index
    const index = stateIndex % this.observationHistory.length
    return this.observationHistory[index] ?? this.model.observation(this._state)
  }

  /**
   * makeTarget -
   * The value target is the discounted root value of the search tree N steps
   * into the future, plus the discounted sum of all rewards until then.
   * @param stateIndex Start index for target values
   * @param numUnrollSteps Number of consecutive game moves to generate target values for
   * @param tdSteps Number of steps in the future to take into account for calculating
   * the target value
   * @param discount Chronological discount of the reward
   */
  public makeTarget (stateIndex: number, numUnrollSteps: number, tdSteps: number, discount: number): MuZeroTarget[] {
    // Convert to positive index
    const index = stateIndex % this.observationHistory.length
    const targets = []
    if (index < this.rootValues.length) {
      targets.push({
        value: this.computeTargetValue(index, tdSteps, discount),
        reward: 0,
        policy: this.childVisits[index]
      })
    } else {
      // States past the end of game are treated as absorbing states.
      targets.push({ value: 0, reward: 0, policy: [] })
    }
    for (let currentIndex = index + 1; currentIndex < index + numUnrollSteps + 1; currentIndex++) {
      if (currentIndex < this.rootValues.length) {
        targets.push({
          value: this.computeTargetValue(currentIndex, tdSteps, discount),
          reward: this.rewards[currentIndex - 1],
          policy: this.childVisits[currentIndex]
        })
      } else {
        // States past the end of game are treated as absorbing states.
        targets.push({ value: 0, reward: 0, policy: [] })
      }
    }
    return targets
  }

  /**
   * computeTargetValue - calculate the discounted root value of the game history at index
   * The value target is the discounted root value of the search tree tdSteps into the
   * future, plus the discounted sum of all rewards until then.
   * @param index
   * @param tdSteps Number of steps in the future to take into account for calculating
   * @param discount Chronological discount of the reward
   * @private
   */
  public computeTargetValue (index: number, tdSteps: number, discount: number): number {
    let value = 0
    // Calculate the discounted root value of the search tree tdSteps into the future
    const bootstrapIndex = index + tdSteps
    if (bootstrapIndex < this.rootValues.length) {
      const samePlayer = this.toPlayHistory[index] === this.toPlayHistory[bootstrapIndex]
      const lastStepValue = samePlayer ? this.rootValues[bootstrapIndex] : -this.rootValues[bootstrapIndex]
      value = lastStepValue * Math.pow(discount, tdSteps)
    }
    // Calculate the discounted sum of all rewards for the tdSteps period
    this.rewards.slice(index, bootstrapIndex + 1).forEach((reward, i) => {
      // The value is oriented from the perspective of the current player
      const samePlayer = this.toPlayHistory[index] === this.toPlayHistory[index + i]
      const r = samePlayer ? reward : -reward
      value += r * Math.pow(discount, i)
    })
    return value
  }

  public historyLength (): number {
    return this.actionHistory.length
  }

  public toString (): string {
    return this.environment.toString(this._state)
  }

  public deserialize (stream: string): Array<MuZeroGameHistory<State, Action>> {
    const games: Array<MuZeroGameHistory<State, Action>> = []
    const objects: MuZeroGameHistoryObject[] = JSON.parse(stream)
    objects.forEach(object => {
      const game = new MuZeroGameHistory(this.environment, this.model)
      object.actionHistory.forEach(oAction => {
        const action = game.legalActions().find(a => a.id === oAction)
        if (action !== undefined) {
          game.apply(action)
        }
      })
      object.rootValues.forEach(r => game.rootValues.push(r))
      object.childVisits.forEach(cv => game.childVisits.push([...cv]))
      object.priorities.forEach(p => game.priorities.push(p))
      game.gamePriority = object.gamePriority
      games.push(game)
    })
    return games
  }

  public serialize (): MuZeroGameHistoryObject {
    return {
      actionHistory: this.actionHistory.map(a => a.id),
      rootValues: this.rootValues,
      childVisits: this.childVisits,
      priorities: this.priorities,
      gamePriority: this.gamePriority
    }
  }
}
