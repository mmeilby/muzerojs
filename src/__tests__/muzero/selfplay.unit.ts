import { describe, expect, test } from '@jest/globals'
import { SharedStorage } from '../../muzero/training/sharedstorage'
import { ReplayBuffer } from '../../muzero/replaybuffer/replaybuffer'
import { SelfPlay } from '../../muzero/selfplay/selfplay'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { MockedNetwork } from '../../muzero/networks/implementations/mocked'
import { ChildNode, type Node, RootNode } from '../../muzero/selfplay/mctsnode'
import { GameHistory } from '../../muzero/selfplay/gamehistory'
import { MuZeroNimUtil } from '../../muzero/games/nim/nimutil'
import debugFactory from 'debug'
import type { Config } from '../../muzero/games/core/config'
import type { Environment } from '../../muzero/games/core/environment'
import { MuZeroNimAction } from '../../muzero/games/nim/nimaction'
import { NetworkState } from '../../muzero/networks/networkstate'

const debug = debugFactory('muzero:unit:debug')

class SelfPlayTest extends SelfPlay {
  constructor (
    conf: Config,
    private readonly fact: Environment
  ) {
    super(conf, fact)
  }

  /**
   * Test selectAction
   * Given the number of visits per child node the node with the highest number of visits must be selected
   * @param visits
   */
  public testSelectAction (visits: number[]): number {
    const root = new RootNode(1, [])
    visits.forEach((v, i) => {
      const child = root.addChild([], new MuZeroNimAction(i))
      child.visits = v
    })
    return this.selectAction(root).id
  }

  public testRunMCTS (gameHistory: GameHistory, network: MockedNetwork): RootNode {
    return this.runMCTS(gameHistory, network)
  }

  public testExpandNode (gameHistory: GameHistory, network: MockedNetwork): RootNode {
    const root = new RootNode(gameHistory.state.player, this.fact.legalActions(gameHistory.state))
    const no = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
    this.expandNode(root, no)
    return root
  }

  public debugChildren (node: Node, index = 0): void {
    debug(`${'.'.repeat(index)} ${node.visits} P${Math.round(node.prior * 100) / 100} V${Math.round(node.value() * 100) / 100} R${Math.round(node.reward * 100) / 100} ${node instanceof ChildNode ? node.action.id : -1}`)
    node.children.forEach(child => {
      //      if (child.visits > 0) {
      this.debugChildren(child, index + 1)
      //      }
    })
  }
}

describe('Muzero Self Play Unit Test:', () => {
  const factory = new MuZeroNim()
  const conf = factory.config()
  // Ensure that SelfPlay ends after only one game production iteration
  conf.trainingSteps = -1
  conf.simulations = 50
  conf.decayingParam = 1.0
  conf.rootExplorationFraction = 0
  conf.pbCbase = 5
  conf.pbCinit = 1.25
  const network = new MockedNetwork(factory)
  const sharedStorage = new SharedStorage(conf, network)
  const selfPlay = new SelfPlay(conf, factory)
  test('Check select action', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const visits = [0, 125, 250, 125, 0, 75, 350, 75, 0]
    const target = 350 // The node with the highest number of visits
    let sucess = 0
    let fails = 0
    for (let i = 0; i < 1000; i++) {
      const action = selfPlayTest.testSelectAction(visits)
      // Don't accept choices with no visits
      if (visits[action] === 0) {
        fails++
      }
      if (visits[action] === target) {
        sucess++
        // Slide the visits to check other positions
        visits.push((visits.shift() ?? 0))
      }
    }
    expect(fails).toEqual(0)
    expect(Math.abs(350 - sucess)).toBeLessThan(35)
  })
  test('Check expand node', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    const no = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
    // save network predicted reward - squeeze to remove batch dimension
    const reward = no.reward
    // save network predicted value - squeeze to remove batch dimension
    const policy = no.policy
    const node = selfPlayTest.testExpandNode(gameHistory, network)
    expect(node.hiddenState?.hiddenState.toString()).toEqual(no.tfHiddenState.toString())
    expect(JSON.stringify(node.hiddenState?.states)).toEqual(JSON.stringify(no.state))
    expect(node.reward).toEqual(reward)
    expect(node.children.length).toEqual(policy.length)
    node.children.forEach(child => {
      expect(child.prior).toEqual(policy[child.action.id])
    })
    expect(node.isExpanded()).toBeTruthy()
  })
  test('Check Monte Carlo Tree Search', async () => {
    const util = new MuZeroNimUtil()
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    let sucess = 0
    for (let i = 0; i < 100; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const top = selfPlayTest.testSelectAction(root.children.map(c => c.visits))
      const topAction = root.children[top].action.id
      // The golden first moves: H1-1, H3-1, H5-1
      if (topAction === 0 || topAction === 3 || topAction === 10) {
        sucess++
      } else {
        debug(`Failed first move: ${root.children[top].action.id} ${util.actionToString(root.children[top].action)} V=${root.children[top].visits}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
        debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
        const badState = factory.step(gameHistory.state, root.children[top].action)
        debug(`Reward for this move: ${factory.reward(badState, gameHistory.state.player)}`)
      }
    }
    expect(sucess).toEqual(100)
  }, 20000)
  test('Check Monte Carlo Tree Search 2', async () => {
    const util = new MuZeroNimUtil()
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    gameHistory.apply(util.actionFromString('H1-1'))
    gameHistory.apply(util.actionFromString('H5-5'))
    gameHistory.apply(util.actionFromString('H4-3'))
    gameHistory.apply(util.actionFromString('H3-1'))
    gameHistory.apply(util.actionFromString('H4-1'))
    gameHistory.apply(util.actionFromString('H3-1'))
    let sucess = 0
    for (let i = 0; i < 100; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const top = selfPlayTest.testSelectAction(root.children.map(c => c.visits))
      const topAction = root.children[top].action.id
      // The golden move: H2-2
      if (topAction === 2) {
        sucess++
      } else {
        debug(`Failed second move: ${root.children[top].action.id} ${util.actionToString(root.children[top].action)} V=${root.children[top].visits}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
        debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
        const badState = factory.step(gameHistory.state, root.children[top].action)
        debug(`Reward for this move: ${factory.reward(badState, gameHistory.state.player)}`)
        //        selfPlayTest.debugChildren(root)
      }
    }
    // Expect at least 90% positive outcome
    expect(sucess).toBeGreaterThan(90)
  }, 10000)
  test('Check Monte Carlo Tree Search SELF PLAY', async () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    for (let i = 0; i < 20; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const top = selfPlayTest.testSelectAction(root.children.map(c => c.visits))
      gameHistory.apply(root.children[top].action)
      debug(`Best move: ${i}: ${gameHistory.state.toString()} V=${root.children[top].visits}`)
      if (gameHistory.terminal()) {
        debug(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
        break
      }
    }
  }, 10000)
  test('Check self play FULL TEST', async () => {
    const replayBuffer = new ReplayBuffer(conf)
    for (let i = 1; i <= 50; i++) {
      await selfPlay.runSelfPlay(sharedStorage, replayBuffer)
    }
    expect(replayBuffer.numPlayedGames).toEqual(50)
    expect(Math.abs(50 - replayBuffer.performance())).toBeLessThanOrEqual(15)
  }, 50000)
})
