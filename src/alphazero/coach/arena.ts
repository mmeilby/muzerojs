import { Config } from '../games/core/config'
import { Environment } from '../games/core/environment'
import { ObservationModel } from '../games/core/model'
import { Statewise } from '../games/core/statewise'
import { Actionwise } from '../games/core/actionwise'

import debugFactory from 'debug'
import { SelfPlay } from '../selfplay/selfplay'
const debug = debugFactory('alphazero:arena:module')

export class ArenaResult {
  public oneWon: number
  public twoWon: number
  public draws: number

  constructor () {
    this.oneWon = 0
    this.twoWon = 0
    this.draws = 0
  }
}

export class Arena<State extends Statewise, Action extends Actionwise> {
  constructor (
    private readonly config: Config,
    private readonly env: Environment<State, Action>,
    private readonly model: ObservationModel<State>,
    private readonly mcts1: SelfPlay<State, Action>,
    private readonly mcts2: SelfPlay<State, Action>
  ) {
  }

  /**
     * playGames
     * Plays num games in which player1 starts num/2 games and player2 starts
     * num/2 games.
     *
     * Returns:
     *   oneWon: games won by player1
     *   twoWon: games won by player2
     *   draws:  games won by nobody
     *
     * @param numGames
     */
  public playGames (numGames: number): ArenaResult {
    const result = new ArenaResult()
    for (let i = Math.floor(numGames / 2); i > 0; i--) {
      const matchResult = this.playGame([this.mcts1, this.mcts2])
      if (matchResult === 1) {
        result.oneWon++
      } else if (matchResult === -1) {
        result.twoWon++
      } else {
        result.draws++
      }
    }
    debug(`NEW/PREV WINS FOR NEW MODEL START : ${result.oneWon} / ${result.twoWon} ; DRAWS : ${result.draws}`)
    for (let i = Math.floor(numGames / 2); i > 0; i--) {
      const matchResult = this.playGame([this.mcts2, this.mcts1])
      if (matchResult === 1) {
        result.twoWon++
      } else if (matchResult === -1) {
        result.oneWon++
      } else {
        result.draws++
      }
    }
    return result
  }

  private playGame (players: Array<SelfPlay<State, Action>>): number {
    let state = this.env.reset()
    while (!this.env.terminal(state)) {
      const player = players.at(state.player < 0 ? 1 : 0)
      if (player !== undefined) {
        const action = player.predictAction(state)
        if (this.env.legalActions(state).find(a => a.id === action.id) !== undefined) {
          state = this.env.step(state, action)
        } else {
          throw new Error(`Invalid move returned from player in Arena: ${this.env.toString(state)} - ${this.env.actionToString(action.id)}`)
        }
      }
    }
    return this.env.reward(state, 1)
  }
}
