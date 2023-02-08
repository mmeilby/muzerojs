import { MuZeroAction } from '../core/action'
import { MuZeroEnvironment } from '../core/environment'
import { MuZeroNimState } from './nimstate'
import { config } from './nimconfig'
import debugFactory from 'debug'

const debug = debugFactory('muzero:nim:module')

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class MuZeroNim implements MuZeroEnvironment<MuZeroNimState, MuZeroAction> {
  /**
   * config
   *  actionSpaceSize number of actions allowed for this game
   *  boardSize number of board positions for this game
   */
  config (): { actionSpaceSize: number, boardSize: number } {
    return {
      actionSpaceSize: config.heaps * config.heapSize,
      boardSize: config.heaps
    }
  }

  public reset (): MuZeroNimState {
    const board: number[] = []
    let pins = config.heapSize
    for (let i = 0; i < config.heaps; i++) {
      board.unshift(config.evenDistributed ? config.heapSize : pins--)
    }
    return new MuZeroNimState(1, board, [])
  }

  public step (state: MuZeroNimState, action: MuZeroAction): MuZeroNimState {
    const boardCopy = [...state.board]
    const heap = Math.floor(action.id / config.heapSize)
    const nimmingSize = action.id % config.heapSize + 1
    boardCopy[heap] = boardCopy[heap] < nimmingSize ? 0 : boardCopy[heap] - nimmingSize
    return new MuZeroNimState(-state.player, boardCopy, state.history.concat([action]))
  }

  public legalActions (state: MuZeroNimState): MuZeroAction[] {
    const legal = []
    for (let i = 0; i < config.heaps; i++) {
      for (let id = 0; id < state.board[i]; id++) {
        legal.push(new MuZeroAction(i * config.heapSize + id))
      }
    }
    return legal
  }

  private haveWinner (state: MuZeroNimState, player?: number): number {
    const ply = player ?? state.player
    const pinsLeft = state.board.reduce((s, p) => s + p, 0)
    if (pinsLeft === 0) {
      return config.misereGame ? ply : -ply
    } else if (pinsLeft === 1) {
      return config.misereGame ? -ply : ply
    } else {
      return 0
    }
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
    const winner = this.haveWinner(state)
    // haveWinner returns the player id of a winning party,
    // so we have to switch the sign if player id is negative
    return winner === 0 ? 0 : winner * player
  }

  public terminal (state: MuZeroNimState): boolean {
    return this.haveWinner(state) !== 0 || this.legalActions(state).length === 0
  }

  public expertAction (state: MuZeroNimState): MuZeroAction {
    const actions = this.legalActions(state)
    const binaryDigitalSum = state.board.reduce((s, p) => s ^ p, 0)
    const maxPinsInHeap = state.board.reduce((s, p) => Math.max(s, p), 0)
    const nonEmptyHeaps = state.board.reduce((s, p) => p > 0 ? s + 1 : s, 0)
    debug(`binaryDigitalSum=${binaryDigitalSum} maxPinsInHeap=${maxPinsInHeap} nonEmptyHeaps=${nonEmptyHeaps}`)
    const scoreTable = []
    for (const action of actions) {
      const heap = Math.floor(action.id / config.heapSize)
      const nimmingSize = action.id % config.heapSize + 1
      if (state.board[heap] >= nimmingSize) {
        const bds = binaryDigitalSum ^ state.board[heap] ^ (state.board[heap] - nimmingSize)
        debug(`${action.id}: ${bds}`)
        const opportunity =
            ((maxPinsInHeap === 1 || nonEmptyHeaps === 1) && config.misereGame && bds === 1) ||
            ((maxPinsInHeap === 1 || nonEmptyHeaps === 1) && !config.misereGame && bds === 0) ||
            (maxPinsInHeap > 1 && nonEmptyHeaps > 1 && bds === 0)
        scoreTable.push({ action, score: opportunity ? Math.random() : -Math.random() })
      }
    }
    scoreTable.sort((a, b) => b.score - a.score)
    return scoreTable.length > 0 ? scoreTable[0].action : new MuZeroAction(-1)
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
    return new MuZeroNimState(player, board, history.map((a: number) => new MuZeroAction(a)))
  }

  public serialize (state: MuZeroNimState): string {
    return JSON.stringify([state.player, state.board, state.history.map(a => a.id)])
  }
}
