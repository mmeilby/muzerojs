import type * as tf from '@tensorflow/tfjs-node-gpu'
import { Action } from '../games/core/action'

/**
 *
 */
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
  public hiddenState: tf.Tensor | undefined
  public readonly children: Node[]

  constructor (
    // Player to take the next action for this state
    public readonly player: number,
    // Possible actions allowed for this state
    public readonly possibleActions: Action[],
    // Action that caused this state - if node is root no action is defined
    public readonly action?: Action | undefined
  ) {
    this.reward = 0
    this.visits = 0
    this.prior = 0
    this.valueSum = 0
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
    const totalVisits = this.children.reduce((sum, child) => sum + child.visits, 0)
    const policy: number[] = new Array(actionSpace).fill(0)
    if (totalVisits !== 0) {
      this.children.forEach(child => {
        if (child.action !== undefined) {
          policy[child.action.id] = child.visits / totalVisits
        }
      })
    }
    return policy
  }

  public addChild (
    possibleActions: Action[],
    action: Action
  ): Node {
    const node = new Node(-this.player, possibleActions, action)
    this.children.push(node)
    return node
  }

  public isExpanded (): boolean {
    return this.possibleActions.length === 0
  }
}
