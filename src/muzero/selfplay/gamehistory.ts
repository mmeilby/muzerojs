import * as tf from '@tensorflow/tfjs-node'
import { type Environment } from '../games/core/environment'
import { type Playerwise } from './entities'
import { type Target } from '../replaybuffer/target'
import { type Action, type Node } from './mctsnode'

interface GameHistoryObject {
  actionHistory: number[]
  childVisits: number[][]
  rootValues: number[]
  priorities: number[]
  gamePriority: number
}

export class GameHistory<State extends Playerwise> {
  private readonly environment: Environment<State> // Game specific environment.
  private readonly actionSpace: number
  private _state: State
  // A list of observation input tensors for the representation network at each turn of the game (before commiting action)
  private readonly observationHistory: tf.Tensor[]
  // A list of actions taken at each turn of the game
  public readonly actionHistory: Action[]
  // A list of true rewards received at each turn of the game related to the player to act
  public readonly rewards: number[]
  // A list of tracked player at each turn of the game (before commiting action)
  public readonly toPlayHistory: number[]
  // A list of action probability distributions for success at each turn of the game
  // Derived from number of visits per root children per action
  public readonly childVisits: number[][]
  // A list of average, discounted rewards expected at each turn of the game
  public readonly rootValues: number[]
  // Root value deviations from true target value - used for priority sorting of saved games
  public priorities: number[]
  // Largest absolute deviation from target value of the priorities list
  public gamePriority: number

  constructor (
    environment: Environment<State>
  ) {
    this.environment = environment
    this.actionSpace = this.environment.config().actionSpace
    this._state = environment.reset()
    this.observationHistory = [this._state.observation]
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
    const state = this.environment.step(this._state, action)
    this.observationHistory.push(state.observation)
    this.rewards.push(this.environment.reward(state, this._state.player))
    this.actionHistory.push(action)
    this.toPlayHistory.push(this._state.player)
    this._state = state
    return state
  }

  public storeSearchStatistics (rootNode: Node<State>): void {
    this.childVisits.push(rootNode.policy(this.actionSpace))
    this.rootValues.push(rootNode.value())
  }

  public makeImage (stateIndex: number): tf.Tensor {
    const image = this.observationHistory.at(stateIndex)
    if (image === undefined) {
      throw new Error(`Invalid index used for makeImage(${stateIndex})`)
    }
    return image
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
   *
   */
  public makeTarget (stateIndex: number, numUnrollSteps: number, tdSteps: number, discount: number): Target[] {
    // Convert to positive index
    const targets: Target[] = []
    for (let currentIndex = stateIndex; currentIndex < stateIndex + numUnrollSteps + 1; currentIndex++) {
      if (currentIndex < this.rootValues.length) {
        targets.push({
          value: tf.tensor2d([[this.computeTargetValue(currentIndex, tdSteps, discount)]]),
          reward: tf.tensor2d([[this.rewards[currentIndex]]]),
          policy: tf.softmax(tf.tensor1d(this.childVisits[currentIndex])).expandDims(0)
        })
      } else {
        // States past the end of game are treated as absorbing states.
        targets.push({
          value: tf.tensor2d([[0]]),
          reward: tf.tensor2d([[0]]),
          policy: tf.tensor2d([[]])
        })
      }
    }
    return targets
  }

  /**
   * computeTargetValue - calculate the discounted root value of the game history at index
   * The value target is the discounted root value of the search tree tdSteps into the
   * future, plus the discounted sum of all rewards until then.
   * @param index Positive index
   * @param tdSteps Number of steps in the future to take into account for calculating
   * @param discount Chronological discount of the reward
   * @private
   */
  public computeTargetValue (index: number, tdSteps: number, discount: number): number {
    let value = 0
    // Calculate the discounted root value of the search tree tdSteps into the future
    const bootstrapIndex = index + tdSteps
    if (bootstrapIndex < this.rootValues.length) {
      //      const samePlayer = this.toPlayHistory[index] === this.toPlayHistory[bootstrapIndex]
      //      const lastStepValue = samePlayer ? this.rootValues[bootstrapIndex] : -this.rootValues[bootstrapIndex]
      value = this.rootValues[bootstrapIndex] * Math.pow(discount, tdSteps)
    }
    // Calculate the discounted sum of all rewards for the tdSteps period
    this.rewards.slice(index, bootstrapIndex + 1).forEach((reward, i) => {
      // The value is oriented from the perspective of the current player
      //    const samePlayer = this.toPlayHistory[index] === this.toPlayHistory[index + i]
      //    const r = samePlayer ? reward : -reward
      value += reward * Math.pow(discount, i)
    })
    return value
  }

  public historyLength (): number {
    return this.actionHistory.length
  }

  public toString (): string {
    return this.environment.toString(this._state)
  }

  public deserialize (stream: string): Array<GameHistory<State>> {
    const games: Array<GameHistory<State>> = []
    const objects: GameHistoryObject[] = JSON.parse(stream)
    objects.forEach(object => {
      const game = new GameHistory(this.environment)
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

  public serialize (): GameHistoryObject {
    return {
      actionHistory: this.actionHistory.map(a => a.id),
      rootValues: this.rootValues,
      childVisits: this.childVisits,
      priorities: this.priorities,
      gamePriority: this.gamePriority
    }
  }
}
