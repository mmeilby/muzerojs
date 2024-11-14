import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from '../games/core/action'
import { type NetworkState } from '../networks/networkstate'
import { type State } from '../games/core/state'

export class Node {
  // The number of times this node has been visited (updated on each back propagation) - N(s,a)
  public visits: number
  // The back propagated value of the node averaged by the number of visits - part of Q(s,a)
  public value: number
  // The predicted reward received by moving to this node - part of Q(s,a)
  public reward: number
  // The discount between the rewards and the values
  public discount: number
  // The predicted value of current state for this node
  public rawValue: number
  // The predicted prior probability of choosing the action that leads to this node - P(s,a)
  public prior: number
  // The hidden state this node corresponds to
  public hiddenState: NetworkState | undefined
  // The game related state this node corresponds to
  public state: State | undefined
  // The possible new states discovered for this node (each child relates to one of the possible actions)
  public readonly children: ChildNode[]

  constructor (
    // Player to take the next action for this state
    public readonly player: number,
    // Possible actions allowed for this state
    public readonly possibleActions: Action[]
  ) {
    this.visits = 0
    this.value = 0
    this.reward = 0
    this.discount = 1
    this.rawValue = 0
    this.prior = 0
    this.children = []
  }

  public qValue (): number {
    return this.reward + this.discount * this.value
  }

  public qValues (actionSpace: number): tf.Tensor1D {
    const indices = tf.tensor1d(this.children.map(child => child.action?.id ?? 0), 'int32')
    const values = tf.tensor1d(this.children.map(child => child.qValue()), 'float32')
    return tf.sparseToDense(indices, values, [actionSpace])
  }

  public samePlayer (player: number): boolean {
    return this.player === player
  }

  policy (actionSpace: number): number[] {
    return tf.tidy(() => {
      const policy = this.childrenVisits(actionSpace)
      return policy.div(policy.sum()).arraySync() as number[]
    })
  }

  childrenVisits (actionSpace: number): tf.Tensor1D {
    const indices = tf.tensor1d(this.children.map(child => child.action?.id ?? 0), 'int32')
    const values = tf.tensor1d(this.children.map(child => child.visits), 'float32')
    return tf.sparseToDense(indices, values, [actionSpace])
  }

  childrenLogits (actionSpace: number): tf.Tensor1D {
    const valueProbs = this.childrenProbs(actionSpace)
    // create virtual logits from inverse Softmax approximation
    const valueLogits = tf.log(tf.where(valueProbs.greater(0), valueProbs.div(valueProbs.max()), 1)) as tf.Tensor1D
    // rescale logits to range [0; 1]
    const min: tf.Tensor1D = valueLogits.min()
    const reduction: tf.Tensor1D = valueLogits.max().sub(min)
    // rescale and return potential logits
    return tf.where(valueProbs.greater(0), valueLogits.sub(min).div(reduction), 0) as tf.Tensor1D
  }

  childrenProbs (actionSpace: number): tf.Tensor1D {
    const indices = tf.tensor1d(this.children.map(child => child.action?.id ?? 0), 'int32')
    const values = tf.tensor1d(this.children.map(child => child.prior), 'float32')
    return tf.sparseToDense(indices, values, [actionSpace])
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

  public disposeHiddenStates (): void {
    this.hiddenState?.hiddenState.dispose()
    for (const child of this.children) {
      child.disposeHiddenStates()
    }
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
