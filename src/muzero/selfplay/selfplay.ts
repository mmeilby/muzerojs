import { type SharedStorage } from '../training/sharedstorage'
import { type ReplayBuffer } from '../replaybuffer/replaybuffer'
import { GameHistory } from './gamehistory'
import { Normalizer } from './normalizer'
import { type Environment } from '../games/core/environment'
import * as tf from '@tensorflow/tfjs-node-gpu'
import debugFactory from 'debug'
import { type Config } from '../games/core/config'
import { type Network } from '../networks/nnet'
import { ChildNode, type Node, RootNode } from './mctsnode'
import { type TensorNetworkOutput } from '../networks/networkoutput'
import { type Action } from '../games/core/action'
import { NetworkState } from '../networks/networkstate'

const info = debugFactory('muzero:selfplay:info')
const log = debugFactory('muzero:selfplay:log')
const debug = debugFactory('muzero:selfplay:debug')

/**
 * MuZeroSelfPlay - where the games are played
 */
export class SelfPlay {
  private readonly actionRange: Action[]

  constructor (
    private readonly config: Config,
    private readonly env: Environment
  ) {
    this.actionRange = env.actionRange()
  }

  /**
   * runSelfPlay - produce multiple games and save them in the shared replay buffer
   * Each self-play job is independent of all others; it takes the latest network
   * snapshot, produces a game and makes it available to the training job by
   * writing it to a shared replay buffer.
   * @param storage
   * @param replayBuffer
   * @param preload
   */
  public async runSelfPlay (storage: SharedStorage, replayBuffer: ReplayBuffer, preload: boolean = false): Promise<void> {
    info('Self play initiated')
    // Produce game plays as long as the training module runs
    do {
      const network = storage.latestNetwork()
      const game = this.playGame(network)
      replayBuffer.saveGame(game)
      log(`Done playing a game with myself - collected ${replayBuffer.numPlayedGames} (${replayBuffer.totalSamples}) samples`)
      log('Played: %s', game.state.toString())
      if (!(preload && replayBuffer.totalGames < this.config.replayBufferSize)) {
        await tf.nextFrame()
      }
    } while (storage.networkCount < this.config.trainingSteps)
    info('Self play completed')
  }

  /**
   * playGame - play a full game using the network to decide the moves
   * Each game is produced by starting at the initial board position, then
   * repeatedly executing a Monte Carlo Tree Search to generate moves until the end
   * of the game is reached.
   * @param network
   * @param index
   * @private
   */
  private playGame (network: Network, index: number = 0): GameHistory {
    const gameHistory = new GameHistory(this.env, this.config)
    // Play a game from start to end, register target data on the fly for the game history
    while (!gameHistory.terminal() && gameHistory.historyLength() < this.config.maxMoves) {
      const rootNode = this.runMCTS(gameHistory, network)
      const action = this.selectAction(rootNode)
      if (debug.enabled) {
        const recommendedAction = this.env.expertAction(gameHistory.state)
        debug(`--- Recommended: ${recommendedAction.id ?? -1}`)
      }
      gameHistory.apply(action)
      gameHistory.storeSearchStatistics(rootNode)
      rootNode.disposeHiddenStates()
      debug(`--- Best action: ${action.id ?? -1} ${gameHistory.state.toString()}`)
    }
    log(`--- STAT(${index}): number of game steps: ${gameHistory.actionHistory.length}`)
    log(`--- VALUES:  ${JSON.stringify(gameHistory.rootValues.map(v => v.toFixed(2)))}`)
    log(`--- REWARD:  ${gameHistory.rewards.toString()} (${gameHistory.rewards.reduce((s, r) => s + r, 0)})`)
    log(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
    return gameHistory
  }

  /**
   * Select the most popular action from the visited nodes
   * @param rootNode The root node to select a child node from
   * @protected
   */
  protected selectAction (rootNode: RootNode): Action {
    let action = 0
    if (rootNode.children.length > 1) {
      tf.tidy(() => {
        // define the probability for each action based on popularity (visits)
        const probs = tf.tensor1d(rootNode.children.map(child => child.visits)).log()
        // select the most popular action - note that for some reason we need to ask for
        // two samples as the first one always is fixed
        action = tf.multinomial(probs, 2).bufferSync().get(1)
      })
    }
    return rootNode.children[action].action
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
  protected runMCTS (gameHistory: GameHistory, network: Network): RootNode {
    const minMaxStats = new Normalizer(this.config.normMin, this.config.normMax)
    const rootNode: Node = new RootNode(gameHistory.state.player, this.env.legalActions(gameHistory.state))
    rootNode.state = gameHistory.state
    tf.tidy(() => {
      // At the root of the search tree we use the representation function to
      // obtain a hidden state given the current observation.
      const tno = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
      this.expandNode(rootNode, tno)
      // Save the hidden state from being disposed by tf.tidy()
      return tno.tfHiddenState
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
        const nodePath: Node[] = [rootNode]
        const debugSearchTree: number[] = []
        while (node.isExpanded()) {
          const selectedNode = this.selectChild(node, minMaxStats)
          nodePath.unshift(selectedNode)
          if (debug.enabled) {
            debugSearchTree.push(selectedNode.action?.id ?? -1)
          }
          node = selectedNode
        }
        debug(`--- MCTS: Simulation: ${sim + 1}: ${debugSearchTree.join('->')}, node probability=${node.prior.toFixed(2)}`)
        // Inside the search tree we use the dynamics function to obtain the next
        // hidden state given an action and the previous hidden state (obtained from parent node).
        if (nodePath.length > 1) {
          const parent = nodePath[1]
          tf.tidy(() => {
            // Ensure some obvious conditions:
            // - that the node is a child node and that the parent hidden state is defined
            if (node instanceof ChildNode) {
              let tno
              if (this.config.supervisedRL) {
                if (parent.state !== undefined) {
                  const state = this.env.step(parent.state, node.action)
                  tno = network.initialInference(new NetworkState(state.observation, [state]))
                  tno.tfReward = tf.scalar(this.env.reward(state, state.player))
                  node.state = state
                } else {
                  // This should not happen - some inconsistency has occurred.
                  throw new Error('Recurrent inference: parent state was unexpectedly undefined')
                }
              } else {
                if (parent.hiddenState !== undefined) {
                  tno = network.recurrentInference(parent.hiddenState, [node.action])
                } else {
                  // This should not happen - some inconsistency has occurred.
                  throw new Error('Recurrent inference: hidden state was unexpectedly undefined')
                }
              }
              this.expandNode(node, tno)
              // Update node path with predicted value - squeeze to remove batch dimension
              this.backPropagate(nodePath, tno.value, node.player, minMaxStats)
              // Save the hidden state from being disposed by tf.tidy()
              return tno.tfHiddenState
            } else {
              // This should not happen - some inconsistency has occurred.
              // Only unexpanded root node does not have hidden state defined
              throw new Error('Recurrent inference: hidden state was unexpectedly undefined')
            }
          })
        } else {
          debug('Recurrent inference attempted on a root node')
          debug(`State: ${gameHistory.state.toString()}`)
          throw new Error('Recurrent inference attempted on a root node. Root node was not fully expanded.')
        }
      }
    }
    return rootNode
  }

  /**
   * expandNode - We expand a node using the value, reward and policy prediction obtained from
   * the neural network.
   * @param node
   * @param networkOutput
   * @private
   */
  protected expandNode (node: Node, networkOutput: TensorNetworkOutput): void {
    // save network predicted hidden state
    node.hiddenState = new NetworkState(networkOutput.tfHiddenState, networkOutput.state)
    // save network predicted reward - squeeze to remove batch dimension
    node.reward = networkOutput.reward
    // save network predicted value - squeeze to remove batch dimension
    const policy = networkOutput.policy
    for (const action of node.possibleActions) {
      // When adding children in the MCTS loop, the possible actions are allowed to be any action
      // The tree search will find the true possible actions
      const child = node.addChild([...this.actionRange], action)
      child.prior = policy[action.id] ?? 0
    }
  }

  /**
   * Add exploration noise
   * At the start of each search, we can add dirichlet noise to the prior of the root
   * to encourage the search to explore new actions.
   * @param rootNode The root node to apply noise to
   * @private
   */
  private addExplorationNoise (rootNode: RootNode): void {
    // Apply noise only if the noise feature is toggled on
    if (this.config.rootExplorationFraction > 0) {
      // make normalized dirichlet noise vector
      const noise = tf.tidy(() => {
        const tfNoise = tf.randomGamma([rootNode.children.length], this.config.rootDirichletAlpha)
        const tfSumNoise = tfNoise.sum()
        return tfNoise.div(tfSumNoise).arraySync() as number[]
      })
      const frac = this.config.rootExplorationFraction
      rootNode.children.forEach((child, i) => {
        child.prior = child.prior * (1 - frac) + noise[i] * frac
      })
    }
  }

  /**
   * Select the child node with the highest UCB score
   * @param node
   * @param minMaxStats
   * @private
   */
  private selectChild (node: Node, minMaxStats: Normalizer): ChildNode {
    if (node.children.length === 0) {
      throw new Error('SelectChild: No children available for selection.')
    }
    if (node.children.length === 1) {
      return node.children[0]
    }
    const ucbTable = node.children.map(child => this.ucbScore(node, child, minMaxStats))
    let bestNode = 0
    let bestUCB = ucbTable[bestNode]
    ucbTable.forEach((ucb, index: number) => {
      if (bestUCB < ucb) {
        bestNode = index
        bestUCB = ucb
      }
    })
    return node.children[bestNode]
  }

  /**
   * ucbScore - The score for a node is based on its value (Q), plus an exploration bonus based
   * on the prior (P - predicted probability of choosing the action that leads to this node)
   *    Upper Confidence Bound
   *    ```U(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))```
   *    Q(s,a): the expected reward for taking action a from state s, i.e. the Q values
   *    N(s,a): the number of times we took action a from state s across simulation
   *    N(s): the sum of all visits at state s across simulation
   *    P(s,a): the initial estimate of taking an action a from the state s according to the
   *    policy returned by the current neural network.
   *    c is a hyperparameter that controls the degree of exploration. The specific choice of c depends on
   *    the problem domain and the trade-off between exploration and exploitation that you want to achieve.
   *    Typically, you might choose c empirically through experimentation or use domain knowledge to guide its selection.
   *    Common values for c are often in the range of 1 to 2, but this can vary based on the problem characteristics and
   *    the desired behavior of the algorithm. If you're unsure about the problem characteristics, or you want to
   *    encourage more initial exploration, you might lean towards larger values of c - like c = 2.
   * The point of the UCB is to initially prefer actions with high prior probability (P) and low visit count (N),
   * but asymptotically prefer actions with a high action value (Q).
   * @param parent
   * @param child
   * @param minMaxStats
   * @param exploit
   * @private
   */
  private ucbScore (parent: Node, child: ChildNode, minMaxStats: Normalizer, exploit = false): number {
    if (exploit) {
      // For exploitation only we simply measure the number of visits
      // Pseudo code actually calculates exp(child.visits / temp) / sum(exp(child.visits / temp))
      return child.visits
    }
    const c = Math.log((parent.visits + this.config.pbCbase + 1) / this.config.pbCbase) + this.config.pbCinit
    const q = child.visits > 0 ? child.reward + child.value() * this.config.decayingParam : 0
    const Q = child.visits > 0 ? minMaxStats.normalize(q) : 0
    if (debug.enabled) {
      if (Math.abs(Q - q) > 0) {
        debug(`Normalize reduced the reward ${Q} from ${q}`)
      }
    }
    return Q + c * child.prior * Math.sqrt(parent.visits) / (1 + child.visits)
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
  private backPropagate (nodePath: Node[], nValue: number, player: number, minMaxStats: Normalizer): void {
    let value = nValue
    for (const node of nodePath) {
      // register visit
      node.visits++
      node.valueSum += node.samePlayer(player) ? value : -value
      debug(`value: ${value} reward: ${node.reward} c-player: ${node.player} player: ${player} c-value: ${node.value()} sum: ${node.valueSum} visits: ${node.visits}`)
      minMaxStats.update(node.value())
      // decay value and include network predicted reward
      value = node.reward + this.config.decayingParam * value
    }
  }
}
