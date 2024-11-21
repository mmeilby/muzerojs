import * as tf from '@tensorflow/tfjs-node-gpu'
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
import { type Action } from '../../muzero/games/core/action'

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
    const root = new RootNode(1, [])
    visits.forEach((v, i) => {
      const child = root.addChild([], new MuZeroNimAction(i))
      child.visits = v
    })
    return this.selectAction(root).id
  }

  public testGumbelSelectActionFromArray (visits: number[]): number {
    const totalVisits = visits.reduce((s, v) => s + v, 0)
    const priors = tf.tidy(() => tf.softmax(tf.tensor1d(visits).div(totalVisits)).arraySync()) as number[]
    const root = new RootNode(1, [])
    visits.forEach((v, i) => {
      const action = new MuZeroNimAction(i)
      root.possibleActions.push(action)
      const child = root.addChild([], action)
      child.visits = v
      child.prior = priors[i]
      child.value = v / totalVisits
      child.discount = this.conf.discount
    })
    return this.gumbelMuzeroInteriorActionSelection(root).id
    //    return this.gumbelMuZeroRootActionSelection(root, this.conf.simulations + 1, 3).id
  }

  public testGumbelRootActionSelection (node: Node): Action {
    return this.gumbelMuZeroRootActionSelection(node, this.conf.simulations + 1, this.conf.actionSpace)
  }

  public testGumbelInteriorSelectAction (node: Node): Action {
    return this.gumbelMuzeroInteriorActionSelection(node)
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

  public testGetTableOfConsideredVisits (maxNumConsideredActions: number, numSimulations: number): number[][] {
    return this.getTableOfConsideredVisits(maxNumConsideredActions, numSimulations)
  }

  public debugChildren (node: Node, index = 0): void {
    debug(`${'.'.repeat(index).padEnd(10, ' ')} ${node.visits.toFixed(0).padStart(2)}V ${(Math.round(node.prior * 100) / 100).toFixed(2).padStart(6)}P ${(Math.round(node.value * 100) / 100).toFixed(2).padStart(6)}V ${(Math.round(node.reward * 100) / 100).toFixed(2).padStart(6)}R ${node instanceof ChildNode ? node.action.toString() : ''}`)
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
  test('Check Node functionality', () => {
    const visits = [0, 125, 250, 125, 0, 75, 350, 75, 0]
    const target = visits.concat(new Array(conf.actionSpace - visits.length).fill(0))
    const totalVisits = visits.reduce((s, v) => s + v, 0)
    const priors = tf.tidy(() => tf.softmax(tf.tensor1d(visits).div(totalVisits)).arraySync()) as number[]
    const root = new RootNode(1, [])
    visits.forEach((v, i) => {
      const action = new MuZeroNimAction(i)
      root.possibleActions.push(action)
      const child = root.addChild([], action)
      child.visits = v
      child.prior = priors[i]
      child.value = v / totalVisits
    })
    expect(root.samePlayer(1)).toBeTruthy()
    expect(root.isExpanded()).toBeTruthy()
    expect(root.policy(conf.actionSpace).map(v => Math.round(v * 1000))).toEqual(target)
    expect(root.childrenVisits(conf.actionSpace).arraySync()).toEqual(target)
    expect(root.childrenProbs(conf.actionSpace).sum().bufferSync().get(0).toFixed(6)).toEqual('1.000000')
    expect(root.childrenLogits(conf.actionSpace).arraySync().map(v => v.toFixed(3))).toEqual(
      ['0.000', '0.357', '0.714', '0.357', '0.000', '0.214', '1.000', '0.214', '0.000', '0.000', '0.000', '0.000', '0.000', '0.000', '0.000']
    )
    expect(root.qValues(conf.actionSpace).arraySync().map(v => Math.round(v * 1000))).toEqual(target)
  })
  test('Check Sequential Halving', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const sequence = selfPlayTest.testGetTableOfConsideredVisits(conf.actionSpace, 10)
    expect(sequence.length).toEqual(conf.actionSpace + 1)
    expect(sequence[0].length).toEqual(10)
  })
  test('Check select action', () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const visits = [0, 125, 250, 125, 0, 75, 350, 75, 0]
    const target = 350 // The node with the highest number of visits
    let sucess = 0
    let gumbelSuccess = 0
    let fails = 0
    let gumbelFails = 0
    for (let i = 0; i < 1000; i++) {
      const action = selfPlayTest.testSelectAction(visits)
      const gumbelAction = selfPlayTest.testGumbelSelectActionFromArray(visits)
      // Don't accept choices with no visits
      if (visits[action] === 0) {
        fails++
      }
      if (visits[gumbelAction] === 0) {
        gumbelFails++
      }
      if (visits[gumbelAction] === target) {
        gumbelSuccess++
      }
      if (visits[action] === target) {
        sucess++
      }
      // Slide the visits to check other positions
      visits.push((visits.shift() ?? 0))
    }
    // Don't expect any choices with no visits
    expect(fails).toEqual(0)
    expect(gumbelFails).toEqual(0)
    // Expect 35% of the choices to be the target - and no more than 10% deviation
    expect(Math.abs(350 - sucess)).toBeLessThan(35)
    // Expect Gumbel to go for the most promising node (visits = 350)
    expect(gumbelSuccess).toEqual(1000)
    debug(`Gumbel success: ${gumbelSuccess} - standard success: ${sucess}`)
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
    for (let i = 0; i < 10; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const topAction = selfPlayTest.testGumbelRootActionSelection(root)
      // The golden first moves: H1-1, H3-1, H5-1
      if (topAction.id === 0 || topAction.id === 3 || topAction.id === 10) {
        if (i === 0) {
          selfPlayTest.debugChildren(root)
          debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
          debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
          const badState = factory.step(gameHistory.state, topAction)
          debug(`Reward for this move: ${factory.reward(badState, gameHistory.state.player)}`)
        }
        sucess++
      } else {
        debug(`Failed first move: ${topAction.id} ${util.actionToString(topAction)} V=${root.children.find(child => child.action.id === topAction.id)?.visits ?? 0}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
        debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
        const badState = factory.step(gameHistory.state, topAction)
        debug(`Reward for this move: ${factory.reward(badState, gameHistory.state.player)}`)
      }
    }
    expect(sucess).toEqual(10)
  })
  test('Check Monte Carlo Tree Search 2', async () => {
    const util = new MuZeroNimUtil()
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    gameHistory.apply(util.actionFromString('H1-1')) // P1
    gameHistory.apply(util.actionFromString('H5-5')) // P2
    gameHistory.apply(util.actionFromString('H4-3')) // P1
    gameHistory.apply(util.actionFromString('H3-1')) // P2
    gameHistory.apply(util.actionFromString('H4-1')) // P1
    gameHistory.apply(util.actionFromString('H3-1')) // P2
    let sucess = 0
    for (let i = 0; i < 10; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      if (i === 0) {
        debug(gameHistory.state.toString())
        selfPlayTest.debugChildren(root)
      }
      const topAction = selfPlayTest.testGumbelInteriorSelectAction(root)
      // The golden move: H2-2
      if (topAction.id === 2) {
        sucess++
      } else {
        debug(`Failed second move: ${topAction.id} ${util.actionToString(topAction)} V=${root.children.find(child => child.action.id === topAction.id)?.visits ?? 0}`)
        debug(`Policy: ${JSON.stringify(root.policy(conf.actionSpace))}`)
        debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
        const badState = factory.step(gameHistory.state, topAction)
        debug(`Reward for this move: ${factory.reward(badState, gameHistory.state.player)}`)
        //        selfPlayTest.debugChildren(root)
      }
    }
    expect(sucess).toEqual(10)
  })
  test('Check Monte Carlo Tree Search SELF PLAY', async () => {
    const selfPlayTest = new SelfPlayTest(conf, factory)
    const gameHistory = new GameHistory(factory, conf)
    for (let i = 0; i < 20; i++) {
      const root = selfPlayTest.testRunMCTS(gameHistory, network)
      const topAction = gameHistory.historyLength() === 0
        ? selfPlayTest.testGumbelRootActionSelection(root)
        : selfPlayTest.testGumbelInteriorSelectAction(root)
      debug(`Best move: ${i}: ${gameHistory.state.toString()}`)
      debug(`Policy: ${root.policy(conf.actionSpace).map(p => p.toFixed(2)).join(', ')}`)
      debug(`Expert advise: A=${factory.expertActionPolicy(gameHistory.state).toString()}`)
      gameHistory.apply(topAction)
      if (gameHistory.terminal()) {
        debug(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
        expect((gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0)).toBeGreaterThan(0)
        break
      }
    }
  })
  test('Check prioritized replay buffer', async () => {
    conf.prioritizedReplay = true
    conf.priorityAlpha = 1.0
    const replayBuffer = new ReplayBuffer(conf)
    await selfPlay.runSelfPlay(sharedStorage, replayBuffer)
    debug(replayBuffer.lastGame?.priorities)
    expect(replayBuffer.lastGame?.gamePriority.toFixed(3)).toEqual('1.256')
  })
  test('Check self play FULL TEST', async () => {
    conf.replayBufferSize = 3
    const replayBuffer = new ReplayBuffer(conf)
    for (let i = 1; i <= 5; i++) {
      await selfPlay.runSelfPlay(sharedStorage, replayBuffer)
    }
    // Expect at least 3 games to be played to fill up the replay buffer
    // The last two iterations may be used for reanalyse of existing games
    expect(replayBuffer.numPlayedGames).toBeGreaterThan(2)
    expect(replayBuffer.performance()).toEqual(100)
  }, 10000)
})
