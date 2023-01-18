import { MuZeroAction } from './core/action'
import * as tf from '@tensorflow/tfjs-node'
import { MuZeroEnvironment } from './core/environment'
import { Playerwise } from '../selfplay/entities'
import { MuZeroModel } from './core/model'

export class MuZeroTicTacToeState implements Playerwise {
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

export class MuZeroTicTacToe implements MuZeroEnvironment<MuZeroTicTacToeState, MuZeroAction> {
  config (): { actionSpaceSize: number, boardSize: number, supportSize: number } {
    return {
      actionSpaceSize: 9,
      boardSize: 9,
      supportSize: 10
    }
  }

  public reset (): MuZeroTicTacToeState {
    return new MuZeroTicTacToeState(1, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [])
  }

  public step (state: MuZeroTicTacToeState, action: MuZeroAction): MuZeroTicTacToeState {
    const boardCopy = state.board.map(r => [...r])
    const row = Math.floor(action.id / 3)
    const col = action.id % 3
    boardCopy[row][col] = state.player
    return new MuZeroTicTacToeState(-state.player, boardCopy, state.history.concat([action]))
  }

  public legalActions (state: MuZeroTicTacToeState): MuZeroAction[] {
    const legal = []
    for (let i = 0; i < 9; i++) {
      const row = Math.floor(i / 3)
      const col = i % 3
      if (state.board[row][col] === 0) {
        legal.push(new MuZeroAction(i))
      }
    }
    return legal
  }

  private static winningPaths (): tf.Tensor[] {
    return [
      tf.oneHot(tf.tensor1d([0, 0, 0], 'int32'), 3),
      tf.oneHot(tf.tensor1d([1, 1, 1], 'int32'), 3),
      tf.oneHot(tf.tensor1d([2, 2, 2], 'int32'), 3),
      tf.tensor2d([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
      tf.tensor2d([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
      tf.tensor2d([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
      tf.oneHot(tf.tensor1d([0, 1, 2], 'int32'), 3),
      tf.oneHot(tf.tensor1d([2, 1, 0], 'int32'), 3)
    ]
  }

  private static haveWinner (state: MuZeroTicTacToeState, player?: number): number {
    const ply = player ?? state.player
    const paths = MuZeroTicTacToe.winningPaths()
    const board = tf.tensor2d(state.board)
    for (const path of paths) {
      const score = board.mul(path).sum().bufferSync().get(0)
      if (Math.abs(score) === 3) {
        return Math.sign(score * ply)
      }
    }
    return 0
  }

  public reward (state: MuZeroTicTacToeState, player: number): number {
    return MuZeroTicTacToe.haveWinner(state, player)
  }

  public terminal (state: MuZeroTicTacToeState): boolean {
    return MuZeroTicTacToe.haveWinner(state) !== 0 || this.legalActions(state).length === 0
  }

  public expertAction (state: MuZeroTicTacToeState): MuZeroAction {
    const paths = MuZeroTicTacToe.winningPaths()
    const actions = this.legalActions(state)
    const scoreTable = []
    for (const action of actions) {
      const boardCopy = state.board.map(r => [...r])
      const row = Math.floor(action.id / 3)
      const col = action.id % 3
      boardCopy[row][col] = state.player
      const board = tf.tensor2d(boardCopy)
      const scores: number[] = []
      for (const path of paths) {
        const score = board.mul(path).sum().bufferSync().get(0)
        scores.push(score * state.player)
      }
      scoreTable.push({ action, score: tf.tensor1d(scores).max().bufferSync().get(0) })
    }
    scoreTable.sort((a, b) => b.score - a.score)
    return scoreTable.length > 0 ? scoreTable[0].action : new MuZeroAction(-1)
  }

  public toString (state: MuZeroTicTacToeState): string {
    const prettyBoard: string[] = []
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
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

  public deserialize (stream: string): MuZeroTicTacToeState {
    const [player, board, history] = JSON.parse(stream)
    return new MuZeroTicTacToeState(player, board, history.map((a: number) => new MuZeroAction(a)))
  }

  public serialize (state: MuZeroTicTacToeState): string {
    return JSON.stringify([state.player, state.board, state.history.map(a => a.id)])
  }
}

export class TicTacToeNetModel implements MuZeroModel<MuZeroTicTacToeState> {
  get observationSize (): number {
    return 27
  }

  public observation (state: MuZeroTicTacToeState): number[][] {
    const boardPlayer1: number[][] = state.board.map(row => row.map(cell => cell === 1 ? 1 : 0))
    const boardPlayer2: number[][] = state.board.map(row => row.map(cell => cell === -1 ? 1 : 0))
    const boardToPlay: number[][] = state.board.map(row => row.map(() => state.player))
    return boardPlayer1.concat(boardPlayer2).concat(boardToPlay)
  }
}
