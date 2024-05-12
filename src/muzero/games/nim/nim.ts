import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Environment } from '../core/environment'
import { MuZeroNimState } from './nimstate'
import { config, util } from './nimconfig'
import debugFactory from 'debug'
import { MuZeroNimUtil } from './nimutil'
import { type Action } from '../../selfplay/mctsnode'
import { Config } from '../core/config'

const debug = debugFactory('muzero:nim:module')

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class MuZeroNim implements Environment<MuZeroNimState> {
  private readonly support = new MuZeroNimUtil()
  private readonly actionSpace = util.heapMap.reduce((s, h) => s + h, 0)

  /**
   * config
   *  actionSpaceSize number of actions allowed for this game
   *  boardSize number of board positions for this game
   */
  config (): Config {
    const conf = new Config(this.actionSpace, new MuZeroNimState(this.actionSpace, [], []).observationSize)
    conf.decayingParam = 0.9
    conf.rootExplorationFraction = 0.1
    conf.pbCbase = 5
    conf.pbCinit = 1.25
    conf.simulations = 50
    conf.lrInit = 0.0001
    conf.lrDecayRate = 0.1
    conf.savedNetworkPath = 'nim'
    return conf
  }

  public reset (): MuZeroNimState {
    const board: number[] = [...util.heapMap]
    return new MuZeroNimState(1, board, [])
  }

  public step (state: MuZeroNimState, action: Action): MuZeroNimState {
    const boardCopy = [...state.board]
    const heap = this.support.actionToHeap(action.id)
    const nimmingSize = this.support.actionToNimming(action.id) + 1
    boardCopy[heap] = boardCopy[heap] < nimmingSize ? 0 : boardCopy[heap] - nimmingSize
    return new MuZeroNimState(-state.player, boardCopy, state.history.concat([action]))
  }

  public legalActions (state: MuZeroNimState): Action[] {
    const legal = []
    // If state is not terminal we can list legal moves
    if (!this.terminal(state)) {
      for (let h = 0; h < config.heaps; h++) {
        for (let n = 0; n < state.board[h]; n++) {
          legal.push(this.support.heapNimmingToAction(h, n))
        }
      }
    }
    return legal
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
  public reward (state: MuZeroNimState, player: number): number {
    // haveWinner returns the player id of a winning party,
    const winner = this.haveWinner(state)
    // so we have to switch the sign if player id is negative
    return winner === 0 ? 0 : winner * player
  }

  public terminal (state: MuZeroNimState): boolean {
    return this.haveWinner(state) !== 0 // || this.legalActions(state).length === 0
  }

  public expertAction (state: MuZeroNimState): Action {
    const scoreTable = this.rankMoves(state)
    if (scoreTable.length > 0) {
      scoreTable.sort((a, b) => b.score - a.score)
      const topActions = scoreTable.filter(st => st.score === scoreTable[0].score).map(st => st.action)
      return topActions[Math.floor(Math.random() * topActions.length)]
    } else {
      return {id: -1}
    }
  }

  public expertActionPolicy (state: MuZeroNimState): tf.Tensor {
    const scoreTable = this.rankMoves(state)
    const actionIds: number[] = scoreTable.map(s => s.action.id)
    const policy: number[] = scoreTable.map(s => s.score)
    return tf.tidy(() => {
      const indices = tf.tensor1d(actionIds, 'int32')
      const values = tf.softmax(tf.tensor1d(policy))
      return tf.sparseToDense(indices, values, [this.actionSpace])
    })
  }

  public toString (state: MuZeroNimState): string {
    const prettyBoard: string[] = []
    for (let i = 0; i < config.heaps; i++) {
      const pinsLeft = state.board[i]
      if (pinsLeft > 0) {
        prettyBoard.push(`${pinsLeft}`)
      } else {
        prettyBoard.push('_')
      }
    }
    return prettyBoard.join('|')
  }

  public deserialize (stream: string): MuZeroNimState {
    const [player, board, history] = JSON.parse(stream)
    return new MuZeroNimState(player, board, history.map((a: number) => {
      return {id: a}
    }))
  }

  public serialize (state: MuZeroNimState): string {
    return JSON.stringify([state.player, state.board, state.history.map(a => a.id)])
  }

  /**
   * haveWinner - return the id of the player winning the game
   * The id returned would be
   *    1 - player 1 wins
   *    -1 - player 2 wins
   *    0 - not possible to claim a winner
   * @param state The state for which the current player will be evaluated
   * @private
   */
  private haveWinner (state: MuZeroNimState): number {
    const pinsLeft = state.board.reduce((s, p) => s + p, 0)
    if (pinsLeft === 0) {
      // The case where previous player has removed the last pin
      // For misére games this defines the current player to win
      return config.misereGame ? state.player : -state.player
    } else if (pinsLeft === 1) {
      // The case when there is only one pin left
      // For misére games this is a loosing situation for the current player
      return config.misereGame ? -state.player : state.player
    } else {
      return 0
    }
  }

  private rankMoves (state: MuZeroNimState): Array<{ action: Action, score: number }> {
    debug(`Move ranking requested for: ${this.toString(state)}`)
    const actions = this.legalActions(state)
    const scoreTable: Array<{ action: Action, score: number }> = []
    for (const action of actions) {
      const heap = this.support.actionToHeap(action.id)
      const nimmingSize = this.support.actionToNimming(action.id) + 1
      if (state.board[heap] >= nimmingSize) {
        const newBoard = [...state.board]
        newBoard[heap] -= nimmingSize
        const binaryDigitalSum = newBoard.reduce((s, p) => s ^ p, 0)
        const maxPinsInHeap = newBoard.reduce((s, p) => Math.max(s, p), 0)
        const nonEmptyHeaps = newBoard.reduce((s, p) => p > 0 ? s + 1 : s, 0)
        debug(`${this.support.actionToString(action)}: bds=${binaryDigitalSum}`)
        debug(`maxPinsInHeap=${maxPinsInHeap} nonEmptyHeaps=${nonEmptyHeaps}`)
        let opportunity = false
        if (config.misereGame) {
          if (maxPinsInHeap === 1) {
            // One stick left in an odd number of heaps is an opportunity
            opportunity = binaryDigitalSum === 1
          } else {
            // General case
            opportunity = nonEmptyHeaps > 1 && binaryDigitalSum === 0
          }
        } else {
          if (maxPinsInHeap === 1) {
            // One stick left in an even number of heaps is an opportunity
            opportunity = binaryDigitalSum === 0
          } else {
            // General case
            opportunity = nonEmptyHeaps !== 1 && binaryDigitalSum === 0
          }
        }
        scoreTable.push({
          action,
          score: opportunity ? 1 : -1
        })
      }
    }
    if (debug.enabled) {
      const scoreTableString = scoreTable.map(st => {
        return `${this.support.actionToString(st.action)}: ${st.score}`
      }).toString()
      debug(`Scoretable: \n${scoreTableString}`)
    }
    return scoreTable
  }
}
