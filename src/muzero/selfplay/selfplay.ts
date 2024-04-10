import { type SharedStorage } from '../training/sharedstorage'
import { type ReplayBuffer } from '../replaybuffer/replaybuffer'
import { GameHistory } from './gamehistory'
import { type Model } from '../games/core/model'
import { Normalizer, type Playerwise } from './entities'
import { type Environment } from '../games/core/environment'
import * as tf from '@tensorflow/tfjs-node'
import debugFactory from 'debug'
import { type NetworkOutput } from '../networks/networkoutput'
import { type Config } from '../games/core/config'
import { type Network } from '../networks/nnet'
import {Action, Node} from "./mctsnode";

/* eslint @typescript-eslint/no-var-requires: "off" */
const { jStat } = require('jstat')
const info = debugFactory('muzero:selfplay:info')
const log = debugFactory('muzero:selfplay:log')
const debug = debugFactory('muzero:selfplay:debug')

/**
 * MuZeroSelfPlay - where the games are played
 */
export class SelfPlay<State extends Playerwise> {
  constructor (
    private readonly config: Config,
    private readonly env: Environment<State>,
    private readonly model: Model<State>
  ) {
  }

  /**
   * runSelfPlay - produce multiple games and save them in the shared replay buffer
   * Each self-play job is independent of all others; it takes the latest network
   * snapshot, produces a game and makes it available to the training job by
   * writing it to a shared replay buffer.
   * @param storage
   * @param replayBuffer
   */
  public async runSelfPlay (storage: SharedStorage, replayBuffer: ReplayBuffer<State>): Promise<void> {
    info(`Self play initiated - running ${this.config.selfPlaySteps} steps`)
    for (let sim = 0; sim < this.config.selfPlaySteps; sim++) {
      const network = storage.latestNetwork()
      const game = this.playGame(network)
      replayBuffer.saveGame(game)
      log(`Done playing a game with myself - collected ${replayBuffer.numPlayedGames} (${replayBuffer.totalSamples}) samples`)
      log('Played: %s', game.state.toString())
      await tf.nextFrame()
    }
    info(`Self play completed - collected ${replayBuffer.totalSamples} samples`)
  }

  /**
   * selfPlay - play a single game and save it in the shared replay buffer
   * @param network
   * @param replayBuffer
   */
  public async selfPlay (network: Network, replayBuffer: ReplayBuffer<State>): Promise<GameHistory<State>> {
    info('Self play initiated - running a single step')
    const game = this.playGame(network)
    replayBuffer.saveGame(game)
    log(`Done playing a game with myself - collected ${game.rootValues.length} samples`)
    log('Played: %s', game.state.toString())
    info(`Total samples collected - ${replayBuffer.totalSamples} samples (${replayBuffer.totalGames}) - ${replayBuffer.numPlayedSteps}/${replayBuffer.numPlayedGames}`)
    return game
  }

  /**
   * playGame - play a full game using the network to decide the moves
   * Each game is produced by starting at the initial board position, then
   * repeatedly executing a Monte Carlo Tree Search to generate moves until the end
   * of the game is reached.
   * @param network
   * @private
   */
  private playGame (network: Network): GameHistory<State> {
    const gameHistory = new GameHistory(this.env, this.model)
    // Play a game from start to end, register target data.old on the fly for the game history
    while (!gameHistory.terminal() && gameHistory.historyLength() < this.config.maxMoves) {
      const rootNode = this.runMCTS(gameHistory, network)
      const action = this.selectAction(rootNode)
      if (debug.enabled) {
        const recommendedAction = this.env.expertAction(gameHistory.state)
        debug(`--- Recommended: ${recommendedAction.id ?? -1}`)
      }
      gameHistory.apply(action)
      gameHistory.storeSearchStatistics(rootNode)
      debug(`--- Best action: ${action.id ?? -1} ${gameHistory.state.toString()}`)
    }
    log(`--- STAT: actions=${gameHistory.actionHistory.length} values=${gameHistory.rootValues.length} rewards=${gameHistory.rewards.length}`)
    log(`--- VALUES:  ${gameHistory.rootValues.toString()}`)
    log(`--- REWARD:  ${gameHistory.rewards.toString()} (${gameHistory.rewards.reduce((s, r) => s + r, 0)})`)
    log(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
    return gameHistory
  }

  /**
   * runMCTS - Core Monte Carlo Tree Search algorithm
   * To decide on an action, we run N simulations, always starting at the root of
   * the search tree and traversing the tree according to the UCB formula until we
   * reach a leaf node.
   * @param gameHistory
   * @param network
   * @private
   */
  private runMCTS (gameHistory: GameHistory<State>, network: Network): Node<State> {
    const minMaxStats = new Normalizer()
    const rootNode: Node<State> = this.createRootNode(gameHistory.state)
    tf.tidy(() => {
      // At the root of the search tree we use the representation function to
      // obtain a hidden state given the current observation.
      const currentObservation = gameHistory.makeImage(-1)
      const networkOutput = network.initialInference(currentObservation)
      //      debug(`Network output: ${JSON.stringify(networkOutput)}`)
      this.expandNode(rootNode, networkOutput)
    })
    // We also need to add exploration noise to the root node actions.
    // This is important to ensure that the Monte Carlo Tree Search explores a range of possible actions
    // rather than only exploring the action which it currently believes to be optimal.
    this.addExplorationNoise(rootNode)
    // We then run a Monte Carlo Tree Search using only action sequences and the
    // model learned by the network.
    if (rootNode.children.length > 0) {
      // If we can make an action for the root node let us descend the tree
      for (let sim = 0; sim < this.config.simulations; sim++) {
        let node = rootNode
        const nodePath: Array<Node<State>> = [rootNode]
        const debugSearchTree: number[] = []
        while (node.isExpanded() && node.children.length > 0) {
          node = this.selectChild(node, minMaxStats)
          nodePath.unshift(node)
          if (debug.enabled) {
            debugSearchTree.push(node.action?.id ?? -1)
          }
        }
        debug(`--- MCTS: Simulation: ${sim + 1}: ${debugSearchTree.join('->')} value=${node.prior}`)
        tf.tidy(() => {
          // Inside the search tree we use the dynamics function to obtain the next
          // hidden state given an action and the previous hidden state (obtained from parent node).
          if (nodePath.length > 1 && node.action !== undefined) {
            //            debug(`Hidden state: ${JSON.stringify(node.parent.mctsState.hiddenState)}`)
            const networkOutput = network.recurrentInference(nodePath[1].hiddenState, node.action)
            this.expandNode(node, networkOutput)
            this.backPropagate(nodePath, networkOutput.nValue, node.player, minMaxStats)
          } else {
            debug('Recurrent inference attempted on a root node')
            debug(`Node: ${JSON.stringify(node.state)}`)
            throw new Error('Recurrent inference attempted on a root node. Root node was not fully expanded.')
          }
        })
      }
    }
    return rootNode
  }

  private createRootNode (state: State): Node<State> {
    // Create new MCTSNode
    return new Node(state, this.env.legalActions(state), state.player)
  }

  private selectAction (rootNode: Node<State>): Action {
    const visitsTable = rootNode.children.map(child => {
      return { action: child.action, visits: child.visits }
    })
    let action = 0
    if (visitsTable.length > 1) {
      // define the probability for each action based on popularity (visits)
      tf.tidy(() => {
        const visitProb = visitsTable.map(r => r.visits)
        const probs = tf.softmax(tf.tensor1d(visitProb))
        // select the most popular action
        action = tf.multinomial(probs, 2).bufferSync().get(1)
      })
    }
    return visitsTable[action].action as Action
  }

  /**
   * randomChoice - make a weighted random choice based on the policy applied
   * @param policy Policy containing a normalized probability vector
   * @returns The action index of the policy with the most probability randomly chosen
   * @protected
   */
  public randomChoice (policy: number[]): number {
    let i = 0
    policy.reduce((s, p) => {
      if (s - p >= 0) {
        i++
      }
      return s - p
    }, Math.random())
    return i
  }

  /**
   * selectChild - Select the child node with the highest UCB score
   * @param node
   * @param minMaxStats
   * @private
   */
  private selectChild (node: Node<State>, minMaxStats: Normalizer): Node<State> {
    if (node.children.length === 0) {
      throw new Error(`SelectChild: No children available for selection. Parent state: ${node.state.toString()}`)
    }
    if (node.children.length === 1) {
      return node.children[0]
    }
    const ucbTable = node.children.map(child => {
      return { node: child, ucb: this.ucbScore(node, child, minMaxStats) }
    })
    const bestUCB = ucbTable.reduce((best, c) => {
      return best.ucb < c.ucb ? c : best
    })
    return bestUCB.node
  }

  /**
   * ucbScore - The score for a node is based on its value, plus an exploration bonus based
   * on the prior (predicted probability of choosing the action that leads to this node)
   *    Upper Confidence Bound
   *    U(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   *    Q(s,a): the expected reward for taking action a from state s, i.e. the Q values
   *    N(s,a): the number of times we took action a from state s across simulation
   *    N(s): the sum of all visits at state s across simulation
   *    P(s,a): the initial estimate of taking an action a from the state s according to the
   *    policy returned by the current neural network.
   *    c is a hyperparameter that controls the degree of exploration
   * The point of the UCB is to initially prefer actions with high prior probability (P) and low visit count (N),
   * but asymptotically prefer actions with a high action value (Q).
   * @param parent
   * @param child
   * @param minMaxStats
   * @param exploit
   * @private
   */
  private ucbScore (parent: Node<State>, child: Node<State>, minMaxStats: Normalizer, exploit = false): number {
    if (exploit) {
      // For exploitation only we simply measure the number of visits
      // Pseudo code actually calculates exp(child.visits / temp) / sum(exp(child.visits / temp))
      return child.visits
    }
    const pbCbase = this.config.pbCbase
    const pbCinit = this.config.pbCinit
    const c = Math.log((parent.visits + pbCbase + 1) / pbCbase) + pbCinit
    const pbC2 = Math.sqrt(parent.visits) / (1 + child.visits)
    const priorScore = c * pbC2 * child.prior
    const valueScore = child.visits > 0 ? minMaxStats.normalize(child.reward + child.value() * this.config.decayingParam) : 0
    if (Number.isNaN(priorScore)) {
      throw new Error('UCB: PriorScore is undefined')
    }
    if (Number.isNaN(valueScore)) {
      debug(`Reward: ${child.reward}, value: ${child.value()}`)
      throw new Error('UCB: ValueScore is undefined')
    }
    return priorScore + valueScore
  }

  /**
   * expandNode - We expand a node using the value, reward and policy prediction obtained from
   * the neural network.
   * @param node
   * @param networkOutput
   * @private
   */
  private expandNode (node: Node<State>, networkOutput: NetworkOutput): void {
    // save network predicted hidden state
    node.hiddenState = networkOutput.aHiddenState
    // save network predicted reward
    node.reward = networkOutput.nReward
    // save network predicted value
    const policySum = networkOutput.policyMap.reduce((p: number, v: number) => p + v, 0)
    let action: Action | undefined
    while ((action = node.possibleActionsLeftToExpand.shift()) !== undefined) {
      const state = this.env.step(node.state, action)
      const child = node.addChild(state, this.env.legalActions(state), action, state.player)
      const p = networkOutput.policyMap[action.id] ?? 0
      child.prior = p / policySum
    }
  }

  /**
   * backPropagate - At the end of a simulation, we propagate the evaluation all the way up the
   * tree to the root.
   * @param nodePath
   * @param nValue
   * @param player
   * @param minMaxStats
   * @private
   */
  private backPropagate (nodePath: Array<Node<State>>, nValue: number, player: number, minMaxStats: Normalizer): void {
    let value = nValue
//    let child: Node<State> | undefined = node
    for (let path = 0; path < nodePath.length; path++) {
      // move to parent node
      const child = nodePath[path]
      if (!Number.isFinite(value)) {
        debug(`nValue: ${nValue} value: ${value}`)
        throw new Error('BackPropagate: Supplied value is out of range')
      }
      // register visit
      child.visits++
      child.valueSum += child.samePlayer(player) ? value : -value
      minMaxStats.update(child.value())
      // decay value and include network predicted reward
      value = child.reward + this.config.decayingParam * value
    }
  }

  // At the start of each search, we add dirichlet noise to the prior of the root
  // to encourage the search to explore new actions.
  // For chess, root_dirichlet_alpha = 0.3
  private addExplorationNoise (node: Node<State>): void {
    // make dirichlet noise vector
    const noise: number[] = []
    let sumNoise = 0
    // first loop with the gamma sample
    for (let i = 0; i < node.children.length; i++) {
      noise[i] = jStat.gamma.sample(this.config.rootDirichletAlpha, node.children.length, 1)
      sumNoise = sumNoise + noise[i]
    }
    // second loop to normalize
    for (let i = 0; i < node.children.length; i++) {
      noise[i] = noise[i] / sumNoise
    }
    const frac = this.config.rootExplorationFraction
    node.children.forEach((child, i) => {
      child.prior = child.prior * (1 - frac) + noise[i] * frac
    })
  }

  /*
    private debugChildren (node: MCTSNode<State, Action>, index = 0): void {
      debug(
        '%s %d P%d V%d R%d %s %o',
        '.'.repeat(index),
        node.mctsState.visits,
        Math.round(node.mctsState.prior * 100) / 100,
        Math.round(node.mctsState.value * 100) / 100,
        Math.round(node.mctsState.reward * 100) / 100,
        node.mctsState.state.toString(),
        node.policy(this.actionSpace_)
      )
    }
  */
}
