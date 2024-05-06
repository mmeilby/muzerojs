import {type SharedStorage} from '../training/sharedstorage'
import {type ReplayBuffer} from '../replaybuffer/replaybuffer'
import {GameHistory} from './gamehistory'
import {Normalizer, type Playerwise} from './entities'
import {type Environment} from '../games/core/environment'
import * as tf from '@tensorflow/tfjs-node'
import debugFactory from 'debug'
import {type Config} from '../games/core/config'
import {type Network} from '../networks/nnet'
import {type Action, Node} from './mctsnode'
import {type TensorNetworkOutput} from '../networks/networkoutput'

/* eslint @typescript-eslint/no-var-requires: "off" */
const {jStat} = require('jstat')
const info = debugFactory('muzero:selfplay:info')
const log = debugFactory('muzero:selfplay:log')
const debug = debugFactory('muzero:selfplay:debug')

/**
 * MuZeroSelfPlay - where the games are played
 */
export class SelfPlay<State extends Playerwise> {
  constructor(
    private readonly config: Config,
    private readonly env: Environment<State>
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
  public async runSelfPlay(storage: SharedStorage, replayBuffer: ReplayBuffer<State>): Promise<void> {
    info('Self play initiated')
    // Produce game plays as long as the training module runs
    while (storage.networkCount < this.config.trainingSteps) {
      // Load the replay buffer with game plays produced from the latest trained network
      await this.preloadReplayBuffer(storage, replayBuffer)
      const certar = this.testNetwork(storage.latestNetwork())
      const certainty = certar.reduce((m, c) => m + c, 0) / certar.length
      log(`--- CERTAINTY: avg = ${certainty.toFixed(2)} - ${JSON.stringify(certar)}`)
      // Wait for the trained network version based on the game plays just added to the replay buffer
      await storage.waitForUpdate()
    }
    info('Self play completed')
  }

  /**
   * preloadReplayBuffer - fill the replay buffer with new current game plays
   * based on the latest version of the trained network
   * @param storage
   * @param replayBuffer
   */
  public async preloadReplayBuffer(storage: SharedStorage, replayBuffer: ReplayBuffer<State>): Promise<void> {
    info(`Preload play initiated - executing ${this.config.replayBufferSize} games`)
    const network = storage.latestNetwork()
    for (let i = 0; i < this.config.replayBufferSize; i++) {
      const game = this.playGame(network)
      replayBuffer.saveGame(game)
    }
    info(`Preload play completed - collected ${replayBuffer.totalSamples} samples`)
  }

  public testNetwork(network: Network): number[] {
    const gameHistory = new GameHistory<State>(this.env)
    const certainty: number[] = []
    const misfit: number[] = []
    // Play a game from start to end, register target data on the fly for the game history
    while (!gameHistory.terminal() && gameHistory.historyLength() < this.config.maxMoves) {
      const currentObservation = gameHistory.makeImage(-1)
      //      tf.tidy(() => {
      const networkOutput = network.initialInference(currentObservation.expandDims(0))
      const legalActions = gameHistory.legalActions().map(a => a.id)
      const policy = networkOutput.tfPolicy.squeeze().arraySync() as number[]
      const legalPolicy: number[] = policy.map((v, i) => legalActions.includes(i) ? v : 0)
      log(`--- Test state: ${gameHistory.state.toString()}`)
      log(`--- Test policy: ${legalPolicy.map(v => v.toFixed(2)).toString()}`)
      const maxProp = legalPolicy.reduce((m, v) => m < v ? v : m, 0)
      const sumProp = policy.reduce((s, v) => s + v, 0)
      const sumPropR = legalPolicy.reduce((s, v) => s + v, 0)
      certainty.push(maxProp / sumProp)
      misfit.push((sumProp - sumPropR) / sumProp)
      const id = legalPolicy.indexOf(maxProp)
      log(`--- Test action: ${id}`)
      gameHistory.apply({id})
      //      })
    }
    log(`--- Test policy: ${gameHistory.state.toString()}`)
    log(`--- Misfit: ${misfit.map(v => v.toFixed(2)).toString()}`)
    return certainty
  }

  /**
   * preloadReplayBuffer - fill the replay buffer with new current game plays
   * based on the latest version of the trained network
   * @param replayBuffer
   */
  public buildTestHistory(replayBuffer: ReplayBuffer<State>): void {
    this.unfoldGame(this.env.reset(), [], replayBuffer)
    info(`Build history completed - generated ${replayBuffer.numPlayedGames} games with ${replayBuffer.totalSamples} samples`)
  }

  /**
   * randomChoice - make a weighted random choice based on the policy applied
   * @param policy Policy containing a normalized probability vector
   * @returns The action index of the policy with the most probability randomly chosen
   * @protected
   */
  public randomChoice(policy: number[]): number {
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
   * playGame - play a full game using the network to decide the moves
   * Each game is produced by starting at the initial board position, then
   * repeatedly executing a Monte Carlo Tree Search to generate moves until the end
   * of the game is reached.
   * @param network
   * @private
   */
  private playGame(network: Network): GameHistory<State> {
    const gameHistory = new GameHistory<State>(this.env)
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
    log(`--- VALUES:  ${JSON.stringify(gameHistory.rootValues.map(v => v.toFixed(2)))}`)
    log(`--- REWARD:  ${gameHistory.rewards.toString()} (${gameHistory.rewards.reduce((s, r) => s + r, 0)})`)
    log(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
    return gameHistory
  }

  private unfoldGame(state: State, actionList: Action[], replayBuffer: ReplayBuffer<State>): void {
    const actions = this.env.legalActions(state)
    for (const action of actions) {
      const newState = this.env.step(state, action)
      if (this.env.terminal(newState)) {
        const gameHistory = new GameHistory<State>(this.env)
        const rootValue = this.env.reward(newState, newState.player)
        for (const a of actionList.concat(action)) {
          const recommendedAction = this.env.expertAction(gameHistory.state)
          const policy = new Array(this.config.actionSpace).fill(0)
          policy[recommendedAction.id] = 1
          gameHistory.rootValues.push(gameHistory.state.player === newState.player ? rootValue : -rootValue)
          gameHistory.childVisits.push(policy)
          gameHistory.apply(a)
        }
        replayBuffer.saveGame(gameHistory)
        //        console.log(JSON.stringify(gameHistory.serialize()))
      } else {
        this.unfoldGame(newState, actionList.concat(action), replayBuffer)
      }
    }
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
  private runMCTS(gameHistory: GameHistory<State>, network: Network): Node<State> {
    const minMaxStats = new Normalizer()
    const rootNode: Node<State> = new Node<State>(gameHistory.state, this.env.legalActions(gameHistory.state))
    tf.tidy(() => {
      // At the root of the search tree we use the representation function to
      // obtain a hidden state given the current observation.
      const currentObservation = gameHistory.makeImage(-1)
      const networkOutput = network.initialInference(currentObservation.expandDims(0))
      this.expandNode(rootNode, networkOutput)
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
          // Inside the search tree we use the dynamics function to obtain the next
          // hidden state given an action and the previous hidden state (obtained from parent node).
          if (nodePath.length > 1 && node.action !== undefined) {
            const tfAction = this.policyTransform(node.action.id)
            const networkOutput = network.recurrentInference(nodePath[1].hiddenState, tfAction)
            debug(`reward: ${networkOutput.tfReward.squeeze().toString()}`)
            this.expandNode(node, networkOutput)
            // Update node path with predicted value - squeeze to remove batch dimension
            this.backPropagate(nodePath, networkOutput.tfValue.squeeze().bufferSync().get(0), node.player, minMaxStats)
          } else {
            debug('Recurrent inference attempted on a root node')
            debug(`Node: ${JSON.stringify(node.state)}`)
            throw new Error('Recurrent inference attempted on a root node. Root node was not fully expanded.')
          }
        }
      }
    })
    return rootNode
  }

  private selectAction(rootNode: Node<State>): Action {
    const visitsTable = rootNode.children.map(child => {
      return {
        action: child.action,
        visits: child.visits
      }
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
   * selectChild - Select the child node with the highest UCB score
   * @param node
   * @param minMaxStats
   * @private
   */
  private selectChild(node: Node<State>, minMaxStats: Normalizer): Node<State> {
    if (node.children.length === 0) {
      throw new Error(`SelectChild: No children available for selection. Parent state: ${node.state.toString()}`)
    }
    if (node.children.length === 1) {
      return node.children[0]
    }
    const ucbTable = node.children.map(child => {
      return {
        node: child,
        ucb: this.ucbScore(node, child, minMaxStats)
      }
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
  private ucbScore(parent: Node<State>, child: Node<State>, minMaxStats: Normalizer, exploit = false): number {
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
  private expandNode(node: Node<State>, networkOutput: TensorNetworkOutput): void {
    // save network predicted hidden state
    node.hiddenState = networkOutput.tfHiddenState
    // save network predicted reward - squeeze to remove batch dimension
    node.reward = networkOutput.tfReward.squeeze().bufferSync().get(0)
    // save network predicted value - squeeze to remove batch dimension
    const policy = networkOutput.tfPolicy.squeeze().arraySync() as number[]
    let action: Action | undefined
    while ((action = node.possibleActionsLeftToExpand.shift()) !== undefined) {
      const state = this.env.step(node.state, action)
      const child = node.addChild(state, this.env.legalActions(state), action)
      child.prior = policy[action.id] ?? 0
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
  private backPropagate(nodePath: Array<Node<State>>, nValue: number, player: number, minMaxStats: Normalizer): void {
    let value = nValue
    for (const node of nodePath) {
      if (!Number.isFinite(value)) {
        debug(`nValue: ${nValue} value: ${value}`)
        throw new Error('BackPropagate: Supplied value is out of range')
      }
      // register visit
      node.visits++
      node.valueSum += node.samePlayer(player) ? value : -value
      debug(`value: ${value} reward: ${node.reward} c-player: ${node.player} player: ${player} c-value: ${node.value()} sum: ${node.valueSum} visits: ${node.visits}`)
      minMaxStats.update(node.value())
      // decay value and include network predicted reward
      value = node.reward + this.config.decayingParam * value
    }
  }

  // At the start of each search, we add dirichlet noise to the prior of the root
  // to encourage the search to explore new actions.
  // For chess, root_dirichlet_alpha = 0.3
  private addExplorationNoise(node: Node<State>): void {
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

  private policyTransform(policy: number): tf.Tensor {
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.config.actionSpace, 1, 0, 'float32')
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
