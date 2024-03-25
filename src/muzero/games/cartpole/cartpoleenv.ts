import { MuZeroAction } from '../core/action'
import { type MuZeroEnvironment } from '../core/environment'
// import debugFactory from 'debug'
import { MuZeroCartpoleState } from './cartpolestate'
import * as tf from '@tensorflow/tfjs-node'

// const debug = debugFactory('muzero:cartpole:module')

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class MuZeroCartpole implements MuZeroEnvironment<MuZeroCartpoleState, MuZeroAction> {
  /**
   * config
   *  actionSpaceSize number of actions allowed for this game
   *  boardSize number of board positions for this game
   */
  config (): { actionSpaceSize: number, boardSize: number } {
    return {
      actionSpaceSize: 2,
      boardSize: 4
    }
  }

  public reset (): MuZeroCartpoleState {
    return new MuZeroCartpoleState(MuZeroCartpoleState.getRandomState(), [])
  }

  public step (state: MuZeroCartpoleState, action: MuZeroAction): MuZeroCartpoleState {
    return new MuZeroCartpoleState(state.update(state.dataset, action.id), state.history.concat([action]))
  }

  /**
   * Get state as a tf.Tensor of shape [1, 4].
   */
  public getObservation (state: MuZeroCartpoleState): tf.Tensor2D {
    return tf.tensor2d([[state.dataset.x, state.dataset.xDot, state.dataset.theta, state.dataset.thetaDot]])
  }

  public legalActions (state: MuZeroCartpoleState): MuZeroAction[] {
    return [new MuZeroAction(0), new MuZeroAction(1)]
  }

  /**
   * Return reward for current state
   * The returned reward would be
   *    1 - for a winning situation
   *    0 - for no current outcome
   *    -1 - for a lost situation
   * @param state
   * @param player
   */
  public reward (state: MuZeroCartpoleState, player: number): number {
    return state.isDone(state.dataset) ? 0 : 1
  }

  public terminal (state: MuZeroCartpoleState): boolean {
    return state.isDone(state.dataset)
  }

  public expertAction (state: MuZeroCartpoleState): MuZeroAction {
    return new MuZeroAction(-1)
  }

  public expertActionPolicy (state: MuZeroCartpoleState): number[] {
    return new Array<number>(this.config().actionSpaceSize).fill(0)
  }

  public toString (state: MuZeroCartpoleState): string {
    return state.toString()
  }

  public deserialize (stream: string): MuZeroCartpoleState {
    const [dataset, history] = JSON.parse(stream)
    return new MuZeroCartpoleState(dataset, history.map((a: number) => new MuZeroAction(a)))
  }

  public serialize (state: MuZeroCartpoleState): string {
    return JSON.stringify([state.dataset, state.history.map(a => a.id)])
  }
}
