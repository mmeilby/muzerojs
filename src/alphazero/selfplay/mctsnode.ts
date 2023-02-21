import { Actionwise } from '../games/core/actionwise'
import { MCTSState } from './mctsstate'

/**
 *
 * @hidden
 * @internal
 * @param State An object representing the state of the game.
 */
export class MCTSNode<State, Action extends Actionwise> {
  private readonly children_: Array<MCTSNode<State, Action>> = []
  private newNode_: boolean = true

  constructor (
    // State related to this node
    private readonly mctsState_: MCTSState<State>,
    // Identification of player to make a move for this state
    private readonly player_: number,
    // Parent node for this node - if node is root then no parent is defined
    private readonly parent_?: MCTSNode<State, Action>,
    // Action that caused this state - if node is root not action is defined
    private readonly action_?: Action
  ) {}

  /**
     * State related to this node
     */
  get mctsState (): MCTSState<State> {
    return this.mctsState_
  }

  /**
     * Identification of player to make a move for this state
     */
  get player (): number {
    return this.player_
  }

  /**
     * Action that caused this state - if node is root no action is defined
     */
  get action (): Action {
    if (this.action_ !== undefined) {
      return this.action_
    } else {
      throw new Error(`MCTS root node has no action defined. node=${JSON.stringify(this.mctsState_)}`)
    }
  }

  get children (): Array<MCTSNode<State, Action>> {
    return this.children_
  }

  /**
     * Parent node for this node - if node is root then no parent is defined, instead the root node is returned
     */
  get parent (): MCTSNode<State, Action> {
    if (this.parent_ !== undefined) {
      return this.parent_
    } else {
      throw new Error(`MCTS root node has no parent defined. node=${JSON.stringify(this.mctsState_)}`)
    }
  }

  get isRootNode (): boolean {
    return this.parent_ === undefined
  }

  policy (actionSpace: number, temp = 0): number[] {
    const probs = Array<number>(actionSpace).fill(0)
    if (temp === 0) {
      // Find the best child node - pick randomly if more child nodes have the same visit count
      const counts: number[] = this.children.map(child => child.mctsState.Nsa)
      const max = counts.reduce((m, v) => Math.max(m, v))
      const bestChildren = this.children.filter(child => child.mctsState.Nsa === max)
      const bestChild = bestChildren[Math.floor(Math.random() * bestChildren.length)]
      probs[bestChild.action.id] = 1
    } else {
      const t = 1 / temp
      this.children.forEach(child => { probs[child.action.id] = Math.pow(child.mctsState.Nsa, t) })
      const sum = probs.reduce((s, v) => s + v)
      if (sum === 0) {
        this.children.forEach(child => { probs[child.action.id] = 1 / this.children.length })
        //                throw new Error(`Sum is zero: props=${probs}, children=${this.children.map(c => c.mctsState.visits)}`)
      } else {
        probs.forEach((p, i) => { probs[i] = p / sum })
      }
    }
    return probs
  }

  addChild (
    mctsState: MCTSState<State>,
    action: Action,
    // Identifcation of player to make a move for this state
    player: number
  ): MCTSNode<State, Action> {
    const node = new MCTSNode(mctsState, player, this, action)
    this.children_.push(node)
    return node
  }

  isNewNode (): boolean {
    return this.newNode_
  }

  isExpanded (): boolean {
    return this.children_.length > 0
  }

  get Nsa (): number {
    return this.mctsState_.Nsa
  }

  set Nsa (value: number) {
    this.mctsState_.Nsa = value
  }

  get Ns (): number {
    return this.parent.mctsState_.Ns
  }

  set Ns (value: number) {
    this.parent.mctsState_.Ns = value
  }

  /**
     * Q(s,a) - The expected reward for taking action a from state s
     */
  get Qsa (): number {
    return this.mctsState_.Qsa
  }

  set Qsa (value: number) {
    this.mctsState_.Qsa = value
  }

  get Psa (): number {
    return this.mctsState_.Psa
  }

  set Psa (value: number) {
    this.mctsState_.Psa = value
  }

  get Es (): number {
    return this.mctsState_.Es
  }

  set Es (value: number) {
    this.mctsState_.Es = value
    this.newNode_ = false
  }
}
