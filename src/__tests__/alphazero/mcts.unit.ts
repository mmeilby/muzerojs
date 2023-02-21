import { describe, test, expect } from '@jest/globals'
import { NimAction } from '../../alphazero/games/nim/nimaction'
import { ReplayBuffer } from '../../alphazero/replaybuffer/replaybuffer'
import { SelfPlay } from '../../alphazero/selfplay/selfplay'
import { Nim } from '../../alphazero/games/nim/nim'
import { NimState } from '../../alphazero/games/nim/nimstate'
import { Config } from '../../alphazero/games/core/config'
import { MockedNetwork } from '../../alphazero/networks/mnetwork'
import { TranspositionTable } from '../../alphazero/selfplay/data-store'
import { NimNetMockedModel } from '../../alphazero/games/nim/nimmmodel'
import { GameHistory } from '../../alphazero/selfplay/gamehistory'
import debugFactory from 'debug'
const debug = debugFactory('muzero:mcts:unit')

describe('MCTS Unit Test:', () => {
  const factory = new Nim()
  const model = new NimNetMockedModel()
  const config = factory.config()
  const conf = new Config(config.actionSpaceSize, model.observationSize)
  test('Check MCTS', async () => {
    const replayBuffer = new ReplayBuffer<NimState, NimAction>(conf)
    const dataStore = new TranspositionTable<NimState>(new Map())
    const network = new MockedNetwork<NimState, NimAction>(factory)
    conf.numEpisodes = 2
    class TestMCTS extends SelfPlay<NimState, NimAction> {
      public testExecuteEpisode (): GameHistory<NimState, NimAction> {
        return this.executeEpisode()
      }
    }
    const mcts = new TestMCTS(conf, factory, model, network)
    const gameHistory = mcts.testExecuteEpisode()
    debug(`--- GAME HISTORY: ${gameHistory.toString()}`)
    debug(`--- PLAYER ${gameHistory.state.player} ${factory.reward(gameHistory.state, gameHistory.state.player) > 0 ? 'wins' : 'looses'}`)
  })
})
