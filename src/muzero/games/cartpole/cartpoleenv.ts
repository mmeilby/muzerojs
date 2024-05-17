import { type Environment } from '../core/environment'
// import debugFactory from 'debug'
import { MuZeroCartpoleState } from './cartpolestate'
import * as tf from '@tensorflow/tfjs-node-gpu'
import { Config } from '../core/config'
import { type Action } from '../core/action'
import { type State } from '../core/state'
import { MuZeroCartpoleAction } from './cartpoleaction'

// const debug = debugFactory('muzero:cartpole:module')

/**
 * Cart pole game implementation
 *
 * For documentation check out:
 * https://gymnasium.farama.org/environments/classic_control/cart_pole/
 */
export class MuZeroCartpole implements Environment {
  private readonly actionSpace = 2

  /**
   * config
   *  actionSpaceSize number of actions allowed for this game
   *  boardSize number of board positions for this game
   */
  config (): Config {
    const conf = new Config(this.actionSpace, new MuZeroCartpoleState({
      x: 0,
      xDot: 0,
      theta: 0,
      thetaDot: 0,
      reward: 0
    }, []).observationShape)
    conf.maxMoves = 500
    conf.decayingParam = 0.997
    conf.rootDirichletAlpha = 0.25
    conf.simulations = 150
    conf.batchSize = 100
    conf.tdSteps = 7
    conf.lrInit = 0.0001
    conf.trainingSteps = 200
    conf.replayBufferSize = 1000
    conf.numUnrollSteps = 500
    conf.lrDecayRate = 0.1
    return conf
  }

  public reset (): MuZeroCartpoleState {
    return new MuZeroCartpoleState(MuZeroCartpoleState.getRandomState(), [])
  }

  public step (state: State, action: Action): MuZeroCartpoleState {
    return new MuZeroCartpoleState(
      (state as MuZeroCartpoleState).update((state as MuZeroCartpoleState).dataset, action.id),
      (state as MuZeroCartpoleState).history.concat([action])
    )
  }

  /**
   * Get state as a tf.Tensor of shape [1, 4].
   */
  public getObservation (state: MuZeroCartpoleState): tf.Tensor2D {
    return tf.tensor2d([[state.dataset.x, state.dataset.xDot, state.dataset.theta, state.dataset.thetaDot]])
  }

  public legalActions (_: State): Action[] {
    return this.actionRange()
  }

  public actionRange (): Action[] {
    return new Array<number>(this.actionSpace).fill(0).map(
      (_, index) => new MuZeroCartpoleAction(index)
    )
  }

  /**
   * Return reward for current state
   * The returned reward would be
   *    1 - for a winning situation
   *    0 - for no current outcome
   *    -1 - for a lost situation
   * @param state
   * @param _
   */
  public reward (state: State, _: number): number {
    return (state as MuZeroCartpoleState).dataset.reward
  }

  public terminal (state: State): boolean {
    return (state as MuZeroCartpoleState).isDone((state as MuZeroCartpoleState).dataset)
  }

  public expertAction (_: State): Action {
    return new MuZeroCartpoleAction()
  }

  public expertActionPolicy (_: State): tf.Tensor {
    return tf.tensor1d(new Array<number>(this.actionSpace).fill(0))
  }

  public toString (state: State): string {
    return state.toString()
  }

  public deserialize (stream: string): MuZeroCartpoleState {
    const [dataset, history] = JSON.parse(stream)
    return new MuZeroCartpoleState(dataset, history.map((a: number) => {
      return new MuZeroCartpoleAction(a)
    }))
  }

  public serialize (state: State): string {
    return JSON.stringify([
      (state as MuZeroCartpoleState).dataset,
      (state as MuZeroCartpoleState).history.map(a => a.id)
    ])
  }
}
