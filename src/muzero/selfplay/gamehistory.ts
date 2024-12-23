import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Environment } from '../games/core/environment'
import { type Target } from '../replaybuffer/target'
import { type RootNode } from './mctsnode'
import { type State } from '../games/core/state'
import { type Action } from '../games/core/action'
import { type Config } from '../games/core/config'

interface GameHistoryObject {
  actionHistory: number[]
  childVisits: number[][]
  rootValues: number[]
  priorities: number[]
  gamePriority: number
}

/**
 * GameHistory - A single episode of interaction with the environment.
 */
export class GameHistory {
  // A list of actions taken at each turn of the game
  public readonly actionHistory: Action[]
  // A list of true rewards received at each turn of the game related to the player to act
  public readonly rewards: number[]
  // A list of tracked player at each turn of the game (before commiting action)
  public readonly toPlayHistory: number[]
  // Derived from number of visits per root children per action
  public readonly childVisits: number[][]
  // A list of average, discounted rewards expected at each turn of the game
  public readonly rootValues: number[]
  // Root value deviations from true target value - used for priority sorting of saved games
  public priorities: number[]
  // Largest absolute deviation from target value of the priorities list
  public gamePriority: number
  // A list of observation input tensors for the representation network at each turn of the game (before commiting action)
  private readonly observationHistory: tf.Tensor[]

  constructor (
    private readonly environment: Environment,
    private readonly config: Config
  ) {
    this._state = environment.reset()
    this.observationHistory = [this._state.observation]
    this.actionHistory = []
    this.rewards = []
    this.toPlayHistory = []
    this.childVisits = []
    this.rootValues = []
    this.priorities = []
    // Preset game priority - recalculate when game history is saved to replay buffer
    this.gamePriority = 1
  }

  private _state: State

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

  /**
   * Return a policy with probabilities only for allowed actions.
   * Illegal actions are assigned the probability of 0.
   * The probabilities are adjusted so the sum is 1.0 even though some actions are masked away.
   * @param policy
   */
  public legalPolicy (policy: number[]): number[] {
    const legalActions = this.legalActions().map(a => a.id)
    const legalPolicy: number[] = policy.map((v, i) => legalActions.includes(i) ? v : 0)
    const psum = legalPolicy.reduce((s, v) => s + v, 0)
    return psum > 0 ? legalPolicy.map(v => v / psum) : legalPolicy
  }

  /**
   * Apply the best action to the environment
   * For each game step the following trajectory is updated:
   *    observationHistory: H(t) - the board state before the action is applied
   *    rewards: R(t+1) - the reward earned from the action in perspective of the player to move
   *    actionHistory: a(t) - the action to apply
   *    toPlayHistory: p(t) - the player to apply the action
   *    childVisits: π(t) - the policy before the action is applied
   *    rootValues: Q(t) - the estimated value of the state before the action is applied
   *    state: s(t) - the next state after applying the action
   * @param action
   */
  public apply (action: Action): State {
    const state = this.environment.step(this._state, action)
    this.observationHistory.push(state.observation)
    this.rewards.push(this.environment.reward(state, this._state.player))
    this.actionHistory.push(action)
    this.toPlayHistory.push(this._state.player)
    this._state = state
    return state
  }

  public storeSearchStatistics (rootNode: RootNode): void {
    this.childVisits.push(rootNode.policy(this.config.actionSpace))
    const value = rootNode.value
    this.rootValues.push(value)
    // Preset priority - recalculate when game history is saved to replay buffer
    this.priorities.push(1)
  }

  public updateSearchStatistics (gameHistoryCopy: GameHistory): void {
    this.childVisits.splice(0, this.childVisits.length, ...gameHistoryCopy.childVisits)
    this.rootValues.splice(0, this.rootValues.length, ...gameHistoryCopy.rootValues)
  }

  public makeImage (stateIndex: number): tf.Tensor {
    const image = this.observationHistory.at(stateIndex)
    if (image === undefined) {
      throw new Error(`Invalid index used for makeImage(${stateIndex})`)
    }
    return image // TODO: make a clone of the image and tidy this like the target values
  }

  /**
   * makeTarget - return a batch of target records for training the network
   * The value target is the discounted root value of the search tree N steps
   * into the future, plus the discounted sum of all rewards until then.
   * @param stateIndex Start index for target values
   * @param numUnrollSteps Number of consecutive game moves to generate target values for
   * @param tdSteps Number of steps in the future to take into account for calculating
   * the target value
   */

  /* Pseudocode
  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets
   */
  public makeTarget (stateIndex: number, numUnrollSteps: number, tdSteps: number): Target[] {
    // def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
    const targets: Target[] = []
    // targets = []
    // const to_play = this._state.player TODO: Why is this relevant as a parameter for make_target in pseudocode? It seems to be the current player for the game history.
    for (let currentIndex = stateIndex; currentIndex < stateIndex + numUnrollSteps + 1; currentIndex++) {
      // # For simplicity the network always predicts the most recently received
      // # reward, even for the initial representation network where we already
      // # know this reward.
      let lastReward = 0
      //   last_reward = 0
      if (currentIndex > 0 && currentIndex <= this.rewards.length) {
        // if current_index > 0 and current_index <= len(self.rewards):
        lastReward = this.rewards[currentIndex - 1]
        //   last_reward = self.rewards[current_index - 1]
      }
      if (currentIndex < this.rootValues.length) {
        // if current_index < len(self.root_values):
        targets.push({
          value: tf.tensor2d([[this.computeTargetValue(currentIndex, tdSteps)]]),
          reward: tf.tensor2d([[lastReward]]),
          policy: tf.tidy(() => tf.softmax(tf.tensor1d(this.childVisits[currentIndex])).expandDims(0))
        })
        // targets.append((value, last_reward, self.child_visits[current_index]))
      } else {
        // States past the end of game are treated as absorbing states.
        targets.push({
          value: tf.tensor2d([[0]]),
          reward: tf.tensor2d([[lastReward]]),
          policy: tf.zeros([1, this.config.actionSpace])
        })
        // targets.append((0, last_reward, []))
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
   * @private
   */
  public computeTargetValue (index: number, tdSteps: number): number {
    let value = 0
    //   value = 0
    // Calculate the discounted root value of the search tree tdSteps into the future
    const bootstrapIndex = index + tdSteps
    // bootstrap_index = current_index + td_steps
    if (bootstrapIndex < this.rootValues.length) {
      // if bootstrap_index < len(self.root_values):
      value = this.rootValues[bootstrapIndex] * Math.pow(this.config.discount, tdSteps)
      //   value = self.root_values[bootstrap_index] * self.discount**td_steps
    }
    // Calculate the discounted sum of all rewards for the tdSteps period
    this.rewards.slice(index, bootstrapIndex).forEach((reward, i) => {
      // for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
      // The value is oriented from the perspective of the current player
      value += reward * Math.pow(this.config.discount, i)
      //   value += reward * self.discount**i
    })
    return value
  }

  public historyLength (): number {
    return this.actionHistory.length
  }

  /**
   * Validate end state for current game.
   * A result of 1 indicates the most desired game play from trained network perspective.
   * All other outcomes are rewarded with 0.
   * The validation can be used to tell how well the network responds.
   * Validation depends on the model, however the parameters used for validation are:
   *    player to take last action
   *    the reward for that action.
   */
  public validateEndState (): number {
    return this.environment.validateReward(this.toPlayHistory.at(-1) ?? 0, this.rewards.at(-1) ?? 0)
  }

  public toString (): string {
    return this._state.toString()
  }

  public deserialize (stream: string): GameHistory[] {
    const actionRange = this.environment.actionRange()
    const games: GameHistory[] = []
    const objects: GameHistoryObject[] = JSON.parse(stream)
    objects.forEach(object => {
      const game = new GameHistory(this.environment, this.config)
      object.actionHistory.forEach(oAction => {
        const action = actionRange[oAction]
        if (action !== undefined && action.id === oAction) {
          game.apply(action)
        } else {
          throw new Error(`Can't parse stream: ${stream}. Invalid action ID: ${oAction}`)
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

  public dispose (): number {
    for (const image of this.observationHistory) {
      image.dispose()
    }
    return this.observationHistory.length
  }
}
