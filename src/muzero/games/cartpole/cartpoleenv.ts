import { type Environment } from '../core/environment'
// import debugFactory from 'debug'
import { MuZeroCartpoleState } from './cartpolestate'
import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from '../../selfplay/mctsnode'
import { Config } from '../core/config'

// const debug = debugFactory('muzero:cartpole:module')

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class MuZeroCartpole implements Environment<MuZeroCartpoleState> {
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
    }, []).observationSize)
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

  public step (state: MuZeroCartpoleState, action: Action): MuZeroCartpoleState {
    return new MuZeroCartpoleState(state.update(state.dataset, action.id), state.history.concat([action]))
  }

  /**
   * Get state as a tf.Tensor of shape [1, 4].
   */
  public getObservation (state: MuZeroCartpoleState): tf.Tensor2D {
    return tf.tensor2d([[state.dataset.x, state.dataset.xDot, state.dataset.theta, state.dataset.thetaDot]])
  }

  public legalActions (_: MuZeroCartpoleState): Action[] {
    return [{id: 0}, {id: 1}]
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
  public reward (state: MuZeroCartpoleState, _: number): number {
    return state.dataset.reward
  }

  public terminal (state: MuZeroCartpoleState): boolean {
    return state.isDone(state.dataset)
  }

  public expertAction (_: MuZeroCartpoleState): Action {
    return {id: -1}
  }

  public expertActionPolicy (_: MuZeroCartpoleState): tf.Tensor {
    return tf.tensor1d(new Array<number>(this.actionSpace).fill(0))
  }

  public toString (state: MuZeroCartpoleState): string {
    return state.toString()
  }

  public deserialize (stream: string): MuZeroCartpoleState {
    const [dataset, history] = JSON.parse(stream)
    return new MuZeroCartpoleState(dataset, history.map((a: number) => {
      return {id: a}
    }))
  }

  public serialize (state: MuZeroCartpoleState): string {
    return JSON.stringify([state.dataset, state.history.map(a => a.id)])
  }
}
