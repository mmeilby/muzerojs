import { MuZeroAction } from './core/action'
import * as tf from '@tensorflow/tfjs-node'
import { MuZeroEnvironment } from './core/environment'
import { Playerwise } from '../selfplay/entities'
import { MuZeroModel } from './core/model'

export class MuZeroConnectFourState implements Playerwise {
  private readonly _key: string
  private readonly _player: number
  private readonly _board: number[][]
  private readonly _history: MuZeroAction[]

  constructor (player: number, board: number[][], history: MuZeroAction[]) {
    this._key = history.length > 0 ? history.map(a => a.id).join(':') : '*'
    this._player = player
    this._board = board
    this._history = history
  }

  get player (): number {
    return this._player
  }

  get board (): number[][] {
    return this._board
  }

  get history (): MuZeroAction[] {
    return this._history
  }

  public toString (): string {
    return this._key
  }
}

export class MuZeroConnectFour implements MuZeroEnvironment<MuZeroConnectFourState, MuZeroAction> {
  config (): { actionSpaceSize: number, boardSize: number, supportSize: number } {
    return {
      actionSpaceSize: 7,
      boardSize: 42,
      supportSize: 10
    }
  }

  public reset (): MuZeroConnectFourState {
    return new MuZeroConnectFourState(1, [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], [])
  }

  public step (state: MuZeroConnectFourState, action: MuZeroAction): MuZeroConnectFourState {
    const boardCopy = state.board.map(r => [...r])
    const row = Math.floor(action.id / 7)
    const col = action.id % 7
    boardCopy[row][col] = state.player
    return new MuZeroConnectFourState(-state.player, boardCopy, state.history.concat([action]))
  }

  public legalActions (state: MuZeroConnectFourState): MuZeroAction[] {
    const legal = []
    for (let col = 0; col < 7; col++) {
      for (let row = 5; row >= 0; row--) {
        if (state.board[row][col] === 0) {
          legal.push(new MuZeroAction(row * 7 + col))
          break
        }
      }
    }
    return legal
  }

  private static winningPaths (): tf.Tensor[] {
    return [
      tf.oneHot(tf.tensor1d([0, 0, 0, 0], 'int32'), 4),
      tf.oneHot(tf.tensor1d([1, 1, 1, 1], 'int32'), 4),
      tf.oneHot(tf.tensor1d([2, 2, 2, 2], 'int32'), 4),
      tf.oneHot(tf.tensor1d([3, 3, 3, 3], 'int32'), 4),
      tf.tensor2d([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
      tf.tensor2d([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
      tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]),
      tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
      tf.oneHot(tf.tensor1d([0, 1, 2, 3], 'int32'), 4),
      tf.oneHot(tf.tensor1d([3, 2, 1, 0], 'int32'), 4)
    ]
  }

  private static haveWinner (state: MuZeroConnectFourState, player?: number): number {
    const ply = player ?? state.player
    const paths = MuZeroConnectFour.winningPaths()
    const board = tf.tensor2d(state.board)
    for (let dcol = 0; dcol < 3; dcol++) {
      for (let drow = 0; drow < 2; drow++) {
        const window = board.slice([drow, dcol], [4, 4])
        for (const path of paths) {
          const score = window.mul(path).sum().bufferSync().get(0)
          if (Math.abs(score) === 4) {
            return Math.sign(score * ply)
          }
        }
      }
    }
    return 0
  }

  public reward (state: MuZeroConnectFourState, player: number): number {
    return MuZeroConnectFour.haveWinner(state, player)
  }

  public terminal (state: MuZeroConnectFourState): boolean {
    return MuZeroConnectFour.haveWinner(state) !== 0 || this.legalActions(state).length === 0
  }

  public expertAction (state: MuZeroConnectFourState): MuZeroAction {
    const paths = MuZeroConnectFour.winningPaths()
    const actions = this.legalActions(state)
    const scoreTable = []
    for (const action of actions) {
      const boardCopy = state.board.map(r => [...r])
      const row = Math.floor(action.id / 7)
      const col = action.id % 7
      boardCopy[row][col] = state.player
      const board = tf.tensor2d(boardCopy)
      const scores: number[] = []
      for (let dcol = 0; dcol < 3; dcol++) {
        for (let drow = 0; drow < 2; drow++) {
          const window = board.slice([drow, dcol], [4, 4])
          for (const path of paths) {
            const score = window.mul(path).sum().bufferSync().get(0)
            scores.push(score * state.player)
          }
        }
      }
      scoreTable.push({ action, score: tf.tensor1d(scores).max().bufferSync().get(0) })
    }
    scoreTable.sort((a, b) => b.score - a.score)
    return scoreTable.length > 0 ? scoreTable[0].action : new MuZeroAction(-1)
  }

  public toString (state: MuZeroConnectFourState): string {
    const prettyBoard: string[] = []
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 7; col++) {
        const typ = state.board[row][col]
        if (typ === 1) {
          prettyBoard.push('X')
        } else if (typ === -1) {
          prettyBoard.push('O')
        } else {
          prettyBoard.push('.')
        }
      }
      prettyBoard.push('\n')
    }
    return prettyBoard.join('')
  }

  public deserialize (stream: string): MuZeroConnectFourState {
    const [player, board, history] = JSON.parse(stream)
    return new MuZeroConnectFourState(player, board, history.map((a: number) => new MuZeroAction(a)))
  }

  public serialize (state: MuZeroConnectFourState): string {
    return JSON.stringify([state.player, state.board, state.history.map(a => a.id)])
  }
}

export class ConnectFourNetModel implements MuZeroModel<MuZeroConnectFourState> {
  get observationSize (): number {
    return 42 * 3
  }

  public observation (state: MuZeroConnectFourState): number[][] {
    const boardPlayer1: number[][] = state.board.map(row => row.map(cell => cell === 1 ? 1 : 0))
    const boardPlayer2: number[][] = state.board.map(row => row.map(cell => cell === -1 ? 1 : 0))
    const boardToPlay: number[][] = state.board.map(row => row.map(() => state.player))
    return boardPlayer1.concat(boardPlayer2).concat(boardToPlay)
  }
}
