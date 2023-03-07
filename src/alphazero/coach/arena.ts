import { Config } from '../games/core/config'
import { Environment } from '../games/core/environment'
import { ObservationModel } from '../games/core/model'
import { Statewise } from '../games/core/statewise'
import { Actionwise } from '../games/core/actionwise'

import debugFactory from 'debug'
import { SelfPlay } from '../selfplay/selfplay'
import {Network} from "../networks/nnet";
import {MockedNetwork} from "../networks/mnetwork";
import {MockedModel} from "../networks/mmodel";
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
    private readonly net1: Network<Action>,
    private readonly net2: Network<Action>
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
  public playGames (numGames: number): ArenaResult[] {
    const playIt = (n: number, networks: Array<Network<Action>>): ArenaResult => {
      const result = new ArenaResult()
      for (let i = n; i > 0; i--) {
        const matchResult = this.playGame(networks)
        if (matchResult > 0) {
          result.oneWon++
        } else if (matchResult < 0) {
          result.twoWon++
        } else {
          result.draws++
        }
      }
      return result
    }
    const result1 = playIt(Math.floor(numGames / 2), [this.net1, this.net2])
    const result2 = playIt(Math.floor(numGames / 2), [this.net2, this.net1])
    return [ result1, result2 ]
  }

  private playGame (networks: Array<Network<Action>>): number {
    const players = networks.map(net => {
      return new SelfPlay(this.config, this.env, net instanceof MockedNetwork<State, Action> ? new MockedModel<State>() : this.model, net)
    })
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
