import * as tf from '@tensorflow/tfjs-node-gpu'
import { TensorNetworkOutput } from '../networkoutput'
import { type Batch } from '../../replaybuffer/batch'
import { type Network } from '../nnet'
import { type Environment } from '../../games/core/environment'
import type { Model } from '../model'
import { type State } from '../../games/core/state'
import type { Action } from '../../games/core/action'
import { NetworkState } from '../networkstate'

/**
 * Mocked network for MuZero reinforced learning
 */
export class MockedNetwork implements Network {
  // Length of the action tensors
  protected readonly actionSpaceN: number
  private readonly actionRange: Action[]

  constructor (
    private readonly env: Environment,
  ) {
    this.actionSpaceN = env.config().actionSpace
    this.actionRange = env.actionRange()
  }

  public initialInference (state: NetworkState): TensorNetworkOutput {
    if (state.states !== undefined) {
      // The mocked network will respond with the perfect move
      const tfValues: tf.Tensor[] = []
      const tfPolicies: tf.Tensor[] = []
      for (const gameState of state.states) {
        tfValues.push(tf.tensor2d([[this.env.reward(gameState, gameState.player)]]))
        tfPolicies.push(this.env.expertActionPolicy(gameState).expandDims(0))
      }
      const tfValue = tf.stack(tfValues)
      const tfPolicy = tf.stack(tfPolicies)
      const tno = new TensorNetworkOutput(tfValue, tf.zerosLike(tfValue), tfPolicy, state.hiddenState)
      tno.state = state.states
      return tno
    } else {
      throw new Error('Game state is undefined for NetworkState')
    }
  }

  public recurrentInference (state: NetworkState, action: Action[]): TensorNetworkOutput {
    if (state.states !== undefined) {
      // The mocked network will respond with the perfect move
      const tfValues: tf.Tensor[] = []
      const tfPolicies: tf.Tensor[] = []
      const newStates: State[] = []
      state.states.forEach((gameState, index) => {
        // @ts-ignore
        const gameAction = action[index]
        const newState = this.env.step(gameState, gameAction)
        newStates.push(newState)
        tfValues.push(tf.tensor2d([[this.env.reward(newState, gameState.player)]]))
        tfPolicies.push(this.env.expertActionPolicy(newState).expandDims(0))
      })
      const tfValue = tf.stack(tfValues)
      const tfPolicy = tf.stack(tfPolicies)
      const tno = new TensorNetworkOutput(tfValue, tfValue, tfPolicy, state.hiddenState)
      tno.state = newStates
      return tno
    } else {
      throw new Error('State is undefined for NetworkState')
    }
  }

  public trainInference (_: Batch[]): number[] {
    // Return the perfect loss and accuracy of 100%
    return [0, 1]
  }

  public getModel (): Model {
    throw new Error('A mocked network has no model to return. GetModel is not implemented.')
  }

  public async save (_: string): Promise<void> {
    // No reason for saving anything from a mocked network
  }

  public async load (_: string): Promise<void> {
    // We can't load any data.old to a mocked network - ignore
  }

  public copyWeights (_: Network): void {
    // A mocked network does not have any data to copy - leave the target network untouched
  }

  public dispose (): number {
    // Nothing to dispose for a mocked network
    return 0
  }
}
