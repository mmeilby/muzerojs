import { NimAction } from './nimaction'
import { Environment } from '../core/environment'
import { NimState } from './nimstate'
import { config } from './nimconfig'
import debugFactory from 'debug'

const debug = debugFactory('muzero:nim:module')

/**
 * NIM game implementation
 *
 * For games history, rules, and theory check out wikipedia:
 * https://en.wikipedia.org/wiki/Nim
 */
export class Nim implements Environment<NimState, NimAction> {
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

  public reset (): NimState {
    const board: number[] = []
    let pins = config.heapSize
    for (let i = 0; i < config.heaps; i++) {
      board.unshift(config.evenDistributed ? config.heapSize : pins--)
    }
    return new NimState(1, board, [])
  }

  public step (state: NimState, action: NimAction): NimState {
    if (action.id < 0) {
      return state
    }
    const boardCopy = [...state.board]
    const heap = Math.floor(action.id / config.heapSize)
    const nimmingSize = action.id % config.heapSize + 1
    boardCopy[heap] = boardCopy[heap] < nimmingSize ? 0 : boardCopy[heap] - nimmingSize
    return new NimState(-state.player, boardCopy, state.history.concat([action]))
  }

  public legalActions (state: NimState): NimAction[] {
    const legal = []
    for (let i = 0; i < config.heaps; i++) {
      for (let id = 0; id < state.board[i]; id++) {
        legal.push(new NimAction(i * config.heapSize + id))
      }
    }
    return legal
  }

  public action (id: number): NimAction {
    return new NimAction(id)
  }

  private haveWinner (state: NimState, player?: number): number {
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
  public reward (state: NimState, player: number): number {
    const winner = this.haveWinner(state)
    // haveWinner returns the player id of a winning party,
    // so we have to switch the sign if player id is negative
    return winner === 0 ? 0 : winner * player
  }

  public terminal (state: NimState): boolean {
    return this.haveWinner(state) !== 0 || this.legalActions(state).length === 0
  }

  public expertAction (state: NimState): number[] {
    debug(`Expert action requested for: ${this.toString(state)}`)
    const actions = this.legalActions(state)
    const scoreTable = []
    for (const action of actions) {
      const heap = Math.floor(action.id / config.heapSize)
      const nimmingSize = action.id % config.heapSize + 1
      if (state.board[heap] >= nimmingSize) {
        const newBoard = [...state.board]
        newBoard[heap] -= nimmingSize
        const binaryDigitalSum = newBoard.reduce((s, p) => s ^ p, 0)
        const maxPinsInHeap = newBoard.reduce((s, p) => Math.max(s, p), 0)
        const nonEmptyHeaps = newBoard.reduce((s, p) => p > 0 ? s + 1 : s, 0)
        debug(`${this.actionToString(action.id)}: bds=${binaryDigitalSum}`)
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
        scoreTable.push({ action, score: opportunity ? 1 : -1 })
      }
    }
    scoreTable.sort((a, b) => b.score - a.score)
    debug(`Scoretable: \n${scoreTable.map(st => {
      return `${this.actionToString(st.action.id)}: ${st.score}`
    })}`)

    const policy: number[] = new Array<number>(this.config().actionSpaceSize).fill(0)
    scoreTable.forEach(s => { policy[s.action.id] = s.score })
    if (policy.every(p => p <= 0)) {
      const sum = policy.reduce((s, p) => s - p)
      return policy.map(p => p < 0 ? 1 / sum : 0)
    } else {
      const sum = policy.filter(p => p > 0).reduce((s, p) => s + p)
      return policy.map(p => p > 0 ? 1 / sum : 0)
    }
  }

  public actionToString (id: number): string {
    if (id < 0) {
      return 'H?-?'
    }
    const heap = Math.floor(id / config.heapSize) + 1
    const nimmingSize = id % config.heapSize + 1
    return `H${heap}-${nimmingSize}`
  }

  public toString (state: NimState): string {
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

  public deserialize (stream: string): NimState {
    const [player, board, history] = JSON.parse(stream)
    return new NimState(player, board, history.map((a: number) => new NimAction(a)))
  }

  public serialize (state: NimState): string {
    return JSON.stringify([state.player, state.board, state.history.map(a => a.id)])
  }
}
