import { MuZeroHiddenState } from '../networks/nnet'

/**
 * `Playerwise` is an interface made to extend generic `State` objects used in
 * the [[GameRules]] interface. It is meant to insure that, even though the shape
 * and implementation of the `State` object is left up to the user, it should
 * atleast have a `player` property.
 */
export interface Playerwise {
  player: number
  toString: () => string
}

export interface Actionwise {
  id: number
}

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
 * @param State An object representing the state of the game.
 * @param Action An object representing an action in the game.
 */
export type GenerateActions<State extends Playerwise, Action> = (state: State) => Action[]

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
 * @param State An object representing the state of the game.
 * @param Action An object representing an action in the game.
 */
export type ApplyAction<State extends Playerwise, Action> = (state: State, action: Action) => State

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
 * @param State An object representing the state of the game.
 */
export type StateIsTerminal<State extends Playerwise> = (state: State) => boolean

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
 * @param State An object representing the state of the game.
 */
export type CalculateReward<State extends Playerwise> = (state: State, player: number) => number

/**
 *
 * @hidden
 * @internal
 * @param State An object representing the state of the game.
 */
export class MCTSNode<State, Action extends Actionwise> {
  private readonly possibleActionsLeftToExpand_: Action[]
  private readonly children_: Array<MCTSNode<State, Action>> = []
  constructor (
    // State related to this node
    private readonly mctsState_: MCTSState<State>,
    // Possible actions allowed for this state
    possibleActions: Action[],
    // Identification of player to make a move for this state
    private readonly player_: number,
    // Parent node for this node - if node is root then no parent is defined
    private readonly parent_?: MCTSNode<State, Action>,
    // Action that caused this state - if node is root not action is defined
    private readonly action_?: Action
  ) {
    this.possibleActionsLeftToExpand_ = possibleActions
  }

  /**
   * State related to this node
   */
  get mctsState (): MCTSState<State> {
    return this.mctsState_
  }

  get possibleActionsLeftToExpand (): Action[] {
    return this.possibleActionsLeftToExpand_
  }

  /**
   * Identification of player to make a move for this state
   */
  get player (): number {
    return this.player_
  }

  /**
   * Action that caused this state - if node is root not action is defined
   */
  get action (): Action | undefined {
    return this.action_
  }

  get children (): Array<MCTSNode<State, Action>> {
    return this.children_
  }

  /**
   * Parent node for this node - if node is root then no parent is defined
   */
  get parent (): MCTSNode<State, Action> | undefined {
    return this.parent_
  }

  policy (actionSpace: number): number[] {
    const totalVisits = this.children.reduce((sum, child) => sum + child.mctsState.visits, 0)
    const policy: number[] = new Array(actionSpace).fill(0)
    if (totalVisits !== 0) {
      this.children.forEach(child => {
        if (child.action !== undefined) {
          policy[child.action.id] = child.mctsState.visits / totalVisits
        }
      })
    }
    return policy
  }

  addChild (
    mctsState: MCTSState<State>,
    possibleActions: Action[],
    action: Action,
    // Identifcation of player to make a move for this state
    player: number
  ): MCTSNode<State, Action> {
    const node = new MCTSNode(mctsState, possibleActions, player, this, action)
    this.children_.push(node)
    return node
  }

  isNotFullyExpanded (): boolean {
    return this.possibleActionsLeftToExpand_.length > 0
  }

  isExpanded (): boolean {
    return this.possibleActionsLeftToExpand_.length === 0
  }
}

/**
 *
 * @hidden
 * @internal
 * @param State An object representing the state of the game.
 */
export class MCTSState<State> {
  // The predicted reward received by moving to this node
  private reward_: number
  // The predicted backfilled value average of the node
  private valueAvg_: number
  // The number of times this node has been visited
  private visits_: number
  // The predicted prior probability of choosing the action that leads to this node
  private prior_: number
  // The backfilled value sum of the node
  private valueSum_: number
  // The hidden state this node corresponds to
  private hiddenState_?: MuZeroHiddenState
  constructor (private readonly state_: State) {
    this.reward_ = 0
    this.visits_ = 0
    this.prior_ = 0
    this.valueSum_ = 0
    this.valueAvg_ = 0
  }

  /**
   * The predicted reward received by moving to this node
   */
  get reward (): number {
    return this.reward_
  }

  set reward (value: number) {
    this.reward_ = value
  }

  /**
   * The number of times this node has been visited (updated on each back propagation)
   */
  get visits (): number {
    return this.visits_
  }

  set visits (value: number) {
    this.visits_ = value
  }

  /**
   * The predicted prior probability of choosing the action that leads to this node
   */
  get prior (): number {
    return this.prior_
  }

  set prior (value: number) {
    this.prior_ = value
  }

  /**
   * The backfilled value sum of the node
   */
  get valueSum (): number {
    return this.valueSum_
  }

  set valueSum (value: number) {
    this.valueSum_ = value
  }

  /**
   * The predicted backfilled value average of the node
   */
  get valueAvg (): number {
    return this.valueAvg_
  }

  set valueAvg (value: number) {
    this.valueAvg_ = value
  }

  /**
   * The backfilled value average by visit of the node
   */
  get value (): number {
    return this.visits_ > 0 ? this.valueSum_ / this.visits_ : 0
  }

  /**
   * The hidden state this node corresponds to
   */
  get hiddenState (): MuZeroHiddenState {
    if (this.hiddenState_ != null) {
      return this.hiddenState_
    }
    throw new Error('Hidden state is undefined for MCTSState')
  }

  set hiddenState (value: MuZeroHiddenState) {
    this.hiddenState_ = value
  }

  get state (): State {
    return this.state_
  }
}

export class Normalizer {
  constructor (
    private min_ = Infinity,
    private max_ = -Infinity
  ) {
  }

  update (value: number): void {
    this.min_ = Math.min(this.min_, value)
    this.max_ = Math.max(this.max_, value)
  }

  normalize (value: number): number {
    return this.max_ > this.min_
      ? (value - this.min_) / (this.max_ - this.min_)
      : value
  }
}
