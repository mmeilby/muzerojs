import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from '../games/core/action'
import { type NetworkState } from '../networks/networkstate'

export class Node {
  // The number of times this node has been visited (updated on each back propagation)
  public visits: number
  // The back propagated value sum of the node
  public valueSum: number
  // The predicted reward received by moving to this node
  public reward: number
  // The predicted prior probability of choosing the action that leads to this node
  public prior: number
  // The hidden state this node corresponds to
  public hiddenState: NetworkState | undefined
  // The possible new states discovered for this node (each child relates to one of the possible actions)
  public readonly children: ChildNode[]

  constructor (
    // Player to take the next action for this state
    public readonly player: number,
    // Possible actions allowed for this state
    public readonly possibleActions: Action[]
  ) {
    this.visits = 0
    this.valueSum = 0
    this.reward = 0
    this.prior = 0
    this.children = []
  }

  /**
   * The back propagated value averaged by visit of the node
   */
  public value (): number {
    return this.visits > 0 ? this.valueSum / this.visits : 0
  }

  public samePlayer (player: number): boolean {
    return this.player === player
  }

  policy (actionSpace: number): number[] {
    return tf.tidy(() => {
      const indices = tf.tensor1d(this.children.map(child => child.action?.id ?? 0), 'int32')
      const values = tf.tensor1d(this.children.map(child => child.visits), 'float32')
      const policy = tf.sparseToDense(indices, values, [actionSpace])
      return policy.div(policy.sum()).arraySync() as number[]
    })
  }

  public addChild (
    possibleActions: Action[],
    action: Action
  ): ChildNode {
    const node = new ChildNode(-this.player, possibleActions, action)
    this.children.push(node)
    return node
  }

  public isExpanded (): boolean {
    return this.children.length > 0
  }
}

/**
 * ```ChildNode```
 * The branch level of the Monte Carlo search tree. The child nodes include a mandatory causing action.
 * */
export class ChildNode extends Node {
  constructor (
    // Player to take the next action for this state
    player: number,
    // Possible actions allowed for this state
    possibleActions: Action[],
    // Action that caused this child state
    public readonly action: Action
  ) {
    super(player, possibleActions)
  }
}

/**
 * ```RootNode```
 * The top level of the Monte Carlo search tree. The root node has no causing action.
 * The root node shares all the attributes from the base class ```Node```
 */
export class RootNode extends Node {
}
