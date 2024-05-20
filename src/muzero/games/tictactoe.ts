import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Environment } from './core/environment'
import { Config } from './core/config'
import { type State } from './core/state'
import { type Action } from './core/action'

export class MuZeroTicTacToeState implements State {
  private readonly _key: string
  private readonly _player: number
  private readonly _board: number[][]
  private readonly _history: Action[]

  constructor (player: number, board: number[][], history: Action[]) {
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

  get history (): Action[] {
    return this._history
  }

  get observationShape (): number[] {
    return [3, 3, 3]
  }

  get observation (): tf.Tensor {
    const boardPlayer1: number[][] = this._board.map(row => row.map(cell => cell === 1 ? 1 : 0))
    const boardPlayer2: number[][] = this._board.map(row => row.map(cell => cell === -1 ? 1 : 0))
    const boardToPlay: number[][] = this._board.map(row => row.map(() => this._player))
    return tf.tensor4d([[boardPlayer1, boardPlayer2, boardToPlay]])
  }

  public toString (): string {
    return this._key
  }
}

export class MuZeroTicTacToeAction implements Action {
  public id: number

  constructor (action?: number) {
    this.id = action ?? -1
  }

  get actionShape (): number[] {
    return [1, 3, 3]
  }

  get action (): tf.Tensor {
    const actionMap: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    const row = Math.floor(this.id / 3)
    const col = this.id % 3
    actionMap[row][col] = 1
    return tf.tensor4d([[actionMap]])
  }

  public toString (): string {
    return this.id.toString()
  }
}

export class MuZeroTicTacToe implements Environment {
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

  private static haveWinner (state: State, player?: number): number {
    const ply = player ?? (state as MuZeroTicTacToeState).player
    const paths = MuZeroTicTacToe.winningPaths()
    const board = tf.tensor2d((state as MuZeroTicTacToeState).board)
    for (const path of paths) {
      const score = board.mul(path).sum().bufferSync().get(0)
      if (Math.abs(score) === 3) {
        return Math.sign(score * ply)
      }
    }
    return 0
  }

  config (): Config {
    const actionSpace = 9
    const conf = new Config(actionSpace, new MuZeroTicTacToeState(actionSpace, [], []).observationShape)
    conf.maxMoves = actionSpace
    conf.decayingParam = 0.997
    conf.rootDirichletAlpha = 0.25
    conf.simulations = 150
    conf.batchSize = 100
    conf.tdSteps = 7
    conf.lrInit = 0.0001
    conf.trainingSteps = 200
    conf.replayBufferSize = 50
    conf.numUnrollSteps = actionSpace
    conf.lrDecayRate = 0.1
    return conf
  }

  public reset (): MuZeroTicTacToeState {
    return new MuZeroTicTacToeState(1, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [])
  }

  public step (state: State, action: Action): MuZeroTicTacToeState {
    const boardCopy = (state as MuZeroTicTacToeState).board.map(r => [...r])
    const row = Math.floor(action.id / 3)
    const col = action.id % 3
    boardCopy[row][col] = (state as MuZeroTicTacToeState).player
    return new MuZeroTicTacToeState(-(state as MuZeroTicTacToeState).player, boardCopy, (state as MuZeroTicTacToeState).history.concat([action]))
  }

  public legalActions (state: State): Action[] {
    const legal = []
    for (let i = 0; i < 9; i++) {
      const row = Math.floor(i / 3)
      const col = i % 3
      if ((state as MuZeroTicTacToeState).board[row][col] === 0) {
        legal.push(new MuZeroTicTacToeAction(i))
      }
    }
    return legal
  }

  public actionRange (): Action[] {
    const legal = []
    for (let i = 0; i < 9; i++) {
      legal.push(new MuZeroTicTacToeAction(i))
    }
    return legal
  }

  public reward (state: State, player: number): number {
    return MuZeroTicTacToe.haveWinner(state, player)
  }

  public terminal (state: State): boolean {
    return MuZeroTicTacToe.haveWinner(state) !== 0 || this.legalActions(state).length === 0
  }

  public expertAction (state: State): Action {
    const paths = MuZeroTicTacToe.winningPaths()
    const actions = this.legalActions(state)
    const scoreTable = []
    for (const action of actions) {
      const boardCopy = (state as MuZeroTicTacToeState).board.map(r => [...r])
      const row = Math.floor(action.id / 3)
      const col = action.id % 3
      boardCopy[row][col] = (state as MuZeroTicTacToeState).player
      const board = tf.tensor2d(boardCopy)
      const scores: number[] = []
      for (const path of paths) {
        const score = board.mul(path).sum().bufferSync().get(0)
        scores.push(score * (state as MuZeroTicTacToeState).player)
      }
      scoreTable.push({
        action,
        score: tf.tensor1d(scores).max().bufferSync().get(0)
      })
    }
    scoreTable.sort((a, b) => b.score - a.score)
    return scoreTable.length > 0 ? scoreTable[0].action : new MuZeroTicTacToeAction()
  }

  public expertActionPolicy (_: State): tf.Tensor {
    return tf.tensor1d(new Array<number>(9).fill(0))
  }

  public toString (state: State): string {
    const prettyBoard: string[] = []
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const typ = (state as MuZeroTicTacToeState).board[row][col]
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
    return new MuZeroTicTacToeState(player, board, history.map((a: number) => {
      return {id: a}
    }))
  }

  public serialize (state: State): string {
    return JSON.stringify([
      (state as MuZeroTicTacToeState).player,
      (state as MuZeroTicTacToeState).board,
      (state as MuZeroTicTacToeState).history.map(a => a.id)
    ])
  }
}
