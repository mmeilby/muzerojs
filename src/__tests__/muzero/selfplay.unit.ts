import { describe, expect, test } from '@jest/globals'
import { SharedStorage } from '../../muzero/training/sharedstorage'
import { ReplayBuffer } from '../../muzero/replaybuffer/replaybuffer'
import { SelfPlay } from '../../muzero/selfplay/selfplay'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { MuZeroNimState } from '../../muzero/games/nim/nimstate'
import { MockedNetwork } from '../../muzero/networks/implementations/mocked'
import { Node } from '../../muzero/selfplay/mctsnode'
import { GameHistory } from '../../muzero/selfplay/gamehistory'
import { MuZeroNimUtil } from '../../muzero/games/nim/nimutil'
import debugFactory from 'debug'
import type { Config } from '../../muzero/games/core/config'
import type { Environment } from '../../muzero/games/core/environment'
import { MuZeroNimAction } from '../../muzero/games/nim/nimaction'

const debug = debugFactory('muzero:unit:debug')

class SelfPlayTest extends SelfPlay {
  constructor (
    private readonly conf: Config,
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
    const root = new Node(1, [])
    visits.forEach((v, i) => {
      const child = root.addChild([], new MuZeroNimAction(i))
      child.visits = v
    })
    return this.selectAction(root).id
  }

  public testRunMCTS (gameHistory: GameHistory, network: MockedNetwork): Node {
    return this.runMCTS(gameHistory, network)
  }

  public testExpandNode (gameHistory: GameHistory, network: MockedNetwork): Node {
    const root = new Node(gameHistory.state.player, this.fact.legalActions(gameHistory.state))
    const no = network.initialInference(gameHistory.makeImage(-1))
    this.expandNode(root, no)
    return root
  }

  public debugChildren (node: Node, index = 0): void {
    debug(`${'.'.repeat(index)} ${node.visits} P${Math.round(node.prior * 100) / 100} V${Math.round(node.value() * 100) / 100} R${Math.round(node.reward * 100) / 100} ${node.action?.id ?? -1}`)
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
  conf.decayingParam = 0.9
  conf.rootExplorationFraction = 0
  conf.pbCbase = 5
  conf.pbCinit = 1.25
  const network = new MockedNetwork(factory, MuZeroNimState.state)
  const sharedStorage = new SharedStorage(conf, network)
  const selfPlay = new SelfPlay(conf, factory)
  test('Check select action', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const visits = [125, 250, 125, 75, 350, 75]
    let target = 4
    let sucess = 0
    for (let i = 0; i < 1000; i++) {
      const action = selfPlayTest.testSelectAction(visits)
      if (action === target) {
        sucess++
        visits.push((visits.shift() ?? 0) + 10)
        target--
        if (target < 0) {
          target = 5
        }
      }
    }
    expect(sucess).toEqual(1000)
  })
  test('Check expand node', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory)
    const no = network.initialInference(gameHistory.makeImage(-1))
    // save network predicted reward - squeeze to remove batch dimension
    const reward = no.tfReward.squeeze().bufferSync().get(0)
    // save network predicted value - squeeze to remove batch dimension
    const policy = no.tfPolicy.squeeze().arraySync() as number[]
    const node = selfPlayTest.testExpandNode(gameHistory, network)
    expect(node.hiddenState?.toString()).toEqual(no.tfHiddenState.toString())
    expect(node.reward).toEqual(reward)
    expect(node.children.length).toEqual(policy.length)
    node.children.forEach(child => {
      expect(child.prior).toEqual(policy[child.action?.id ?? 0])
    })
    expect(node.isExpanded()).toBeTruthy()
  })
  test('Check Monte Carlo Tree Search', async () => {
    const util = new MuZeroNimUtil()
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory)
    let sucess = 0
    for (let i = 0; i < 100; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const top = selfPlayTest.testSelectAction(root.children.map(c => c.visits))
      const topAction = root.children[top].action?.id ?? -1
      // The golden first moves: H1-1, H3-1, H5-1
      if (topAction === 0 || topAction === 3 || topAction === 10) {
        sucess++
      } else {
        debug(`Failed first move: ${root.children[top].action?.id ?? -1} ${util.actionToString(root.children[top].action ?? new MuZeroNimAction())} V=${root.children[top].visits}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
      }
    }
    expect(sucess).toEqual(100)
  }, 20000)
  test('Check Monte Carlo Tree Search 2', async () => {
    const util = new MuZeroNimUtil()
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory)
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
      const topAction = root.children[top].action?.id
      // The golden move: H2-2
      if (topAction === 2) {
        sucess++
      } else {
        debug(`Failed second move: ${root.children[top].action?.id ?? -1} ${util.actionToString(root.children[top].action ?? new MuZeroNimAction())} V=${root.children[top].visits}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
        debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
        selfPlayTest.debugChildren(root)
      }
    }
    expect(sucess).toEqual(100)
  }, 10000)
  test('Check Monte Carlo Tree Search SELF PLAY', async () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory)
    for (let i = 0; i < 20; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const top = selfPlayTest.testSelectAction(root.children.map(c => c.visits))
      const topAction = root.children[top].action ?? new MuZeroNimAction()
      gameHistory.apply(topAction)
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
    expect(replayBuffer.statistics()).toEqual(100)
  }, 50000)
})
