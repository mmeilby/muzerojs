import { Environment } from '../games/core/environment'
import { ObservationModel } from '../games/core/model'
import { Target } from '../replaybuffer/target'
import { Observation } from '../networks/nnet'
import { Actionwise } from '../games/core/actionwise'
import { Statewise } from '../games/core/statewise'
import { HistoryObject } from './history'

interface GameHistoryObject {
  actionHistory: number[]
  childVisits: number[][]
  rootValues: number[]
  priorities: number[]
  gamePriority: number
}

export class GameHistory<State extends Statewise, Action extends Actionwise> {
  private readonly history: Array<HistoryObject<State, Action>>
  private readonly environment: Environment<State, Action> // Game specific environment.
  private readonly model: ObservationModel<State>
  private _state: State
  // A list of actions taken at each turn of the game
  //  public readonly actionHistory: Action[]
  // A list of true rewards received at each turn of the game
  //  public readonly rewards: number[]
  // A list of tracked player at each turn of the game
  //  public readonly toPlayHistory: number[]
  // A list of action probability distributions from the root node at each turn of the game
  // Derived from number of visits per root children per action
  //  public readonly childVisits: number[][]
  // A list of values of the root node at each turn of the game
  //  public readonly rootValues: number[]
  // Root value deviations from true target value - used for priority sorting of saved games
  //  public priorities: number[]
  // Largest absolute deviation from target value of the priorities list
  //  public gamePriority: number

  constructor (
    environment: Environment<State, Action>,
    model: ObservationModel<State>
  ) {
    this.environment = environment
    this.model = model
    this._state = environment.reset()
    this.history = []
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
    this.history.push(new HistoryObject(this._state, action))
    const state = this.environment.step(this._state, action)
    this._state = state
    return state
  }

  public storeSearchStatistics (pi: number[], value: number): void {
    const historyObject = this.history.at(-1)
    if (historyObject !== undefined) {
      historyObject.pi = pi
      historyObject.value = value
    }
  }

  public updateRewards (): void {
    const player = this._state.player
    const reward = this.environment.reward(this._state, player) * (player !== 1 ? -1 : 1)
    this.history.forEach((historyObject, i) => {
      historyObject.value = reward * Math.pow(-1, i)
    })
  }

  public makeImage (stateIndex: number): Observation {
    // Game specific feature planes.
    const historyObject = this.history.at(stateIndex)
    if (historyObject !== undefined) {
      return this.model.observation(historyObject.state)
    } else {
      return this.model.observation(this._state)
    }
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
  public makeTarget (stateIndex: number, numUnrollSteps: number, tdSteps: number, discount: number): Target[] {
    // Convert to positive index
    const index = stateIndex % this.history.length
    const targets = []
    for (let currentIndex = index; currentIndex < index + numUnrollSteps; currentIndex++) {
      targets.push({
        value: this.computeTargetValue(currentIndex, tdSteps, discount),
        reward: this.history.at(currentIndex)?.value ?? 0,
        policy: this.history.at(currentIndex)?.pi ?? []
      })
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
    const bootstrapHistoryObject = this.history.at(bootstrapIndex)
    const historyObject = this.history.at(index)
    if (bootstrapHistoryObject !== undefined && historyObject !== undefined) {
      const samePlayer = historyObject.player === bootstrapHistoryObject.player
      const lastStepValue = samePlayer ? bootstrapHistoryObject.value : -bootstrapHistoryObject.value
      value = lastStepValue * Math.pow(discount, tdSteps)
      // Calculate the discounted sum of all rewards for the tdSteps period
      this.history.map(h => h.value).slice(index, bootstrapIndex + 1).forEach((reward, i) => {
        // The value is oriented from the perspective of the current player
        const historyObjectI = this.history.at(index + i)
        const samePlayer = historyObject.player === historyObjectI?.player
        const r = samePlayer ? reward : -reward
        value += r * Math.pow(discount, i)
      })
    }
    return value
  }

  public episodeStep (): number {
    return this.history.length + 1
  }

  public recordedSteps (): number {
    return this.history.length
  }

  public toString (): string {
    return this.history.map(h => {
      return `State: ${this.environment.toString(h.state)}, ` +
        `action: ${this.environment.actionToString(h.action.id)} => ` +
        `value: ${h.value}`
    }).join('; ')
  }

  public deserialize (stream: string): Array<GameHistory<State, Action>> {
    const games: Array<GameHistory<State, Action>> = []
    const objects: GameHistoryObject[] = JSON.parse(stream)
    objects.forEach((object, i) => {
      const game = new GameHistory(this.environment, this.model)
      object.actionHistory.forEach(oAction => {
        const action = game.legalActions().find(a => a.id === oAction)
        if (action !== undefined) {
          game.apply(action)
          game.storeSearchStatistics(object.childVisits.at(i) ?? [], object.rootValues.at(i) ?? 0)
        }
      })
      games.push(game)
    })
    return games
  }

  public serialize (): GameHistoryObject {
    return {
      actionHistory: this.history.map(h => h.action.id),
      rootValues: this.history.map(h => h.value),
      childVisits: this.history.map(h => h.pi),
      priorities: [],
      gamePriority: 0
    }
  }
}
