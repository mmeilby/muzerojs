import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Environment } from '../../muzero/games/core/environment'
import { Config } from '../../muzero/games/core/config'
import { type State } from '../../muzero/games/core/state'
import { type Action } from '../../muzero/games/core/action'

const actionSpace = 5

/**
 * Unit test game implementation
 *
 * This is a simple game with 5 slots til fill. The center slot is common for player 1 and 2
 * The rest are available for each player as the following:
 * *---*---*---*---*---*
 * | 1 | 1 | C | 2 | 2 |
 * *---*---*---*---*---*
 * When both private slots are full the center slot is available. First player to fill center slot wins.
 */
export class GameTest implements Environment {
  /**
   * config
   *  actionSpaceSize number of actions allowed for this game
   *  boardSize number of board positions for this game
   */
  config (): Config {
    const conf = new Config(
      actionSpace,
      new GameTestState(actionSpace, [], []).observationShape,
      new GameTestAction(0).actionShape
    )
    conf.decayingParam = 1.0
    conf.rootExplorationFraction = 0.25
    conf.pbCbase = 50
    conf.pbCinit = 1.25
    conf.simulations = 50
    conf.savedNetworkPath = 'gt'
    conf.normMin = -1
    conf.normMax = 1
    return conf
  }

  public reset (): GameTestState {
    const board: number[] = new Array(actionSpace).fill(0)
    return new GameTestState(1, board, [])
  }

  public step (state: State, action: Action): GameTestState {
    const gameTestState = this.castState(state)
    if (action instanceof GameTestAction) {
      const boardCopy = [...gameTestState.board]
      boardCopy[action.id] = 1
      return new GameTestState(-gameTestState.player, boardCopy, gameTestState.history.concat([action]))
    }
    throw new Error('Action is not instance of GameTestAction')
  }

  public legalActions (state: State): GameTestAction[] {
    const gameTestState = this.castState(state)
    const legal: GameTestAction[] = []
    // If state is not terminal we can list legal moves
    if (!this.terminal(gameTestState)) {
      if (gameTestState.player > 0) {
        if (gameTestState.board[3] > 0 && gameTestState.board[4] > 0) {
          legal.push(new GameTestAction(2))
        }
        legal.push(new GameTestAction(3))
        legal.push(new GameTestAction(4))
      } else {
        legal.push(new GameTestAction(0))
        legal.push(new GameTestAction(1))
        if (gameTestState.board[0] > 0 && gameTestState.board[1] > 0) {
          legal.push(new GameTestAction(2))
        }
      }
    }
    return legal
  }

  public actionRange (): GameTestAction[] {
    return new Array<number>(actionSpace).fill(0).map(
      (_, index) => new GameTestAction(index)
    )
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
  public reward (state: State, player: number): number {
    const gameTestState = this.castState(state)
    if (gameTestState.board[2] > 0) {
      return gameTestState.player * player
    }
    return 0
  }

  public validateReward (player: number, reward: number): number {
    const winner = player * reward
    return winner > 0 ? 1 : 0
  }

  public terminal (state: State): boolean {
    const gameTestState = this.castState(state)
    return gameTestState.board[2] > 0
  }

  public expertAction (state: State): Action {
    const gameTestState = this.castState(state)
    const index = gameTestState.player > 0 ? 3 : 0
    const sum = gameTestState.board.slice(index, index + 2).reduce((s: number, v: number) => s + v, 0)
    switch (sum) {
      case 0: {
        return new GameTestAction(index)
      }
      case 1: {
        return gameTestState.board[index] > 0 ? new GameTestAction(index + 1) : new GameTestAction(index)
      }
      case 2: {
        return new GameTestAction(2)
      }
    }
    return new GameTestAction()
  }

  public expertActionPolicy (state: State): tf.Tensor {
    const gameTestState = this.castState(state)
    const index = gameTestState.player > 0 ? 3 : 0
    const sum = gameTestState.board.slice(index, index + 2).reduce((s: number, v: number) => s + v, 0)
    const policy: number[] = new Array(actionSpace).fill(0)
    switch (sum) {
      case 0: {
        policy[index] = 0.5
        policy[index + 1] = 0.5
        break
      }
      case 1: {
        policy[index] = gameTestState.board[index] > 0 ? 0 : 1
        policy[index + 1] = gameTestState.board[index + 1] > 0 ? 0 : 1
        break
      }
      case 2: {
        policy[2] = 1
        break
      }
    }
    return tf.tidy(() => {
      const indices = tf.tensor1d([0, 1, 2, 3, 4], 'int32')
      const values = tf.softmax(tf.tensor1d(policy))
      return tf.sparseToDense(indices, values, [actionSpace])
    })
  }

  public toString (state: State): string {
    const gameTestState = this.castState(state)
    const prettyBoard: string[] = []
    for (let i = 0; i < actionSpace; i++) {
      if (gameTestState.board[i] > 0) {
        prettyBoard.push('X')
      } else {
        prettyBoard.push('_')
      }
    }
    return prettyBoard.join('|')
  }

  public deserialize (stream: string): GameTestState {
    const [player, board, history] = JSON.parse(stream)
    return new GameTestState(player, board, history.map((a: number) => {
      return new GameTestAction(a)
    }))
  }

  public serialize (state: State): string {
    const gameTestState = this.castState(state)
    return JSON.stringify([gameTestState.player, gameTestState.board, gameTestState.history.map((a: GameTestAction) => a.id)])
  }

  private castState (state: State): GameTestState {
    if (state instanceof GameTestState) {
      return state
    }
    throw new Error('State is not instance of NimState')
  }
}

export class GameTestAction implements Action {
  public id: number

  constructor (action?: number) {
    this.id = action ?? -1
  }

  get actionShape (): number[] {
    return [actionSpace, 1, 1]
  }

  get action (): tf.Tensor {
    const board: number[] = new Array(actionSpace).fill(0)
    board[this.id] = 1
    return tf.tensor3d([[board]])
  }

  public toString (): string {
    if (this.id < 0) {
      return 'H?'
    }
    return `H${this.id - 3}`
  }
}

export class GameTestState implements State {
  constructor (
    public readonly player: number,
    public readonly board: number[],
    public readonly history: Action[]
  ) {
  }

  /**
   * Return the shape of the observation tensors excluding the batch dimension
   */
  get observationShape (): number[] {
    return [actionSpace, 1, 1]
  }

  /**
   * Make an observation tensor shaped as observationShape including the batch dimension
   */
  get observation (): tf.Tensor {
    return tf.tensor3d([[this.board]])
  }

  public toString (): string {
    const actionHistory = this.history.length > 0 ? this.history.map(a => a.id).join(':') : '*'
    const actionStringHistory = this.history.length > 0 ? this.history.map(a => a.toString()).join(',') : '*'
    return `${actionHistory} | ${actionStringHistory} | ${this.board.join('-')}`
  }
}
