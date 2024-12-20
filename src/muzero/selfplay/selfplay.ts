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
const test = debugFactory('muzero:selfplay:test')
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
   */
  public async runSelfPlay (storage: SharedStorage, replayBuffer: ReplayBuffer): Promise<void> {
    info('Self play initiated')
    // Produce game plays or reanalyse existing game plays as long as the training module runs
    do {
      const network = storage.latestNetwork()
      // Only reanalyse if replay buffer is full
      if (replayBuffer.totalGames === this.config.replayBufferSize && Math.random() >= this.config.reanalyseFactor) {
        const gameHistory = replayBuffer.games.at(storage.networkCount % replayBuffer.totalGames)
        if (gameHistory !== undefined) {
          const gameHistoryCopy = this.reanalyse(network, gameHistory)
          gameHistory.updateSearchStatistics(gameHistoryCopy)
          // Clean up intermediate tensors used for reanalyse
          gameHistoryCopy.dispose()
          // If game priorities are used then reset for new root values
          replayBuffer.setGamePriority(gameHistory)
        }
      } else {
        const game = this.playGame(network)
        replayBuffer.saveGame(game)
        log(`Done playing a game with myself - collected ${replayBuffer.numPlayedGames} (${replayBuffer.totalSamples}) samples`)
        log('Played: %s', game.state.toString())
      }
      // Allow training agent to continue in next time frame
      await tf.nextFrame()
    } while (storage.networkCount < this.config.trainingSteps)
    info('Self play completed')
  }

  /**
   * Decide the best move for current state
   * The method uses Monte Carlo Tree Search to support the network policy responses
   * @param gameHistory Game history to track the state and progress
   * @param network Trained network to qualify decisions
   * @returns The best Action to do based on the current state
   */
  public decide (gameHistory: GameHistory, network: Network): Action {
    const rootNode = this.runMCTS(gameHistory, network)
    const action = gameHistory.historyLength() === 0
      ? this.gumbelMuZeroRootActionSelection(rootNode, this.config.simulations + 1, this.config.actionSpace)
      : this.gumbelMuzeroInteriorActionSelection(rootNode)
    rootNode.disposeHiddenStates()
    return action
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
      //      const action = this.selectAction(rootNode)
      const action = gameHistory.historyLength() === 0
        ? this.gumbelMuZeroRootActionSelection(rootNode, this.config.simulations + 1, this.config.actionSpace)
        : this.gumbelMuzeroInteriorActionSelection(rootNode)
      if (test.enabled) {
        const recommendedAction = this.env.expertAction(gameHistory.state)
        test(`--- Recommended: ${recommendedAction.id ?? -1}`)
      }
      gameHistory.apply(action)
      gameHistory.storeSearchStatistics(rootNode)
      rootNode.disposeHiddenStates()
      test(`--- Best action: ${action.id ?? -1} ${gameHistory.state.toString()}`)
    }
    log(`--- STAT(${index}): number of game steps: ${gameHistory.actionHistory.length}`)
    log(`--- VALUES:  ${JSON.stringify(gameHistory.rootValues.map(v => v.toFixed(2)))}`)
    log(`--- REWARD:  ${gameHistory.rewards.toString()} (${gameHistory.rewards.reduce((s, r) => s + r, 0)})`)
    log(`--- WINNER: ${(gameHistory.toPlayHistory.at(-1) ?? 0) * (gameHistory.rewards.at(-1) ?? 0) > 0 ? '1' : '2'}`)
    return gameHistory
  }

  /**
   * reanalyse - replay a full game using the network to update policy and value targets
   * Each game is produced by starting at the initial board position, then
   * repeatedly executing a Monte Carlo Tree Search to generate updates until the end
   * of the game is reached.
   * @param network
   * @param gameHistory
   * @private
   */
  private reanalyse (network: Network, gameHistory: GameHistory): GameHistory {
    const gameHistoryCopy = new GameHistory(this.env, this.config)
    // Reanalyse a game from start to end, register target data on the fly for the game history
    for (const action of gameHistory.actionHistory) {
      const rootNode = this.runMCTS(gameHistoryCopy, network)
      gameHistoryCopy.apply(action)
      gameHistoryCopy.storeSearchStatistics(rootNode)
      rootNode.disposeHiddenStates()
    }
    log(`--- NEW VALUES:  ${JSON.stringify(gameHistoryCopy.rootValues.map(v => v.toFixed(2)))}`)
    return gameHistoryCopy
  }

  /**
   * Select the most popular action from the visited nodes
   * @param rootNode The root node to select a child node from
   * @protected
   */
  protected selectAction (rootNode: RootNode): Action {
    let action = 0
    if (rootNode.children.length > 1) {
      const visits = rootNode.children.map(child => child.visits)
      const results = new Array(visits.length).fill(0)
      const sum = visits.reduce((sum, visit) => sum + visit)
      for (let t = 0; t < 10; t++) {
        const random = Math.random() * sum
        let cumulative = 0
        for (let i = 0; i < visits.length; i++) {
          cumulative += visits[i]
          if (random < cumulative) {
            results[i]++
            break
          }
        }
      }
      action = this.maxArg(results)
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
          const selectedNode = this.selectChild(node)
          nodePath.unshift(selectedNode)
          if (test.enabled) {
            debugSearchTree.push(selectedNode.action?.id ?? -1)
          }
          node = selectedNode
        }
        test(`--- MCTS: Simulation: ${sim + 1}: ${debugSearchTree.join('->')}, node probability=${node.prior.toFixed(2)}`)
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
                  tno.tfReward = tf.scalar(this.env.reward(state, parent.player))
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
              // adjust reward for root node perspective - if root node player and parent player are not the same reverse the reward
              if (!rootNode.samePlayer(parent.player)) {
                node.reward *= -1
              }
              // Update node path with predicted value - squeeze to remove batch dimension
              this.backPropagate(nodePath, tno.value)
              // Save the hidden state from being disposed by tf.tidy()
              return tno.tfHiddenState
            } else {
              // This should not happen - some inconsistency has occurred.
              // Only unexpanded root node does not have hidden state defined
              throw new Error('Recurrent inference: hidden state was unexpectedly undefined')
            }
          })
        } else {
          test('Recurrent inference attempted on a root node')
          test(`State: ${gameHistory.state.toString()}`)
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
    // save network predicted value (used by Q transformation with mix values)
    node.rawValue = networkOutput.value
    // configure value discount for Q-value
    node.discount = this.config.decayingParam
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
   * backPropagate - At the end of a simulation, we propagate the evaluation all the way up the
   * tree to the root.
   * @param nodePath queue of nodes traversed from the root before finding the node to be expanded
   * @param rawValue the network predicted long-term prospect from the current node onward
   * @private
   */
  protected backPropagate (nodePath: Node[], rawValue: number): void {
    let value = rawValue
    for (const node of nodePath) {
      // register visit
      node.visits++
      // The following adds the value to the mean value
      // Could also be expressed the following way before node.visits++:
      //    node.value = (node.value * node.visits + value) / (node.visits + 1)
      node.value += (value - node.value) / node.visits
      // decay value and include network predicted reward
      value = node.reward + this.config.decayingParam * value
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
   * @private
   */
  private selectChild (node: Node): ChildNode {
    if (node.children.length === 0) {
      throw new Error('SelectChild: No children available for selection.')
    }
    if (node.children.length === 1) {
      return node.children[0]
    }
    const minMaxStats = new Normalizer(this.config.normMin, this.config.normMax)
    const initQValue = this.calculateInitQValue(node, minMaxStats)
    const ucbTable = node.children.map(child => this.ucbScore(node.visits, child, minMaxStats, initQValue))
    const bestNode = this.maxArg(ucbTable)
    return node.children[bestNode]
  }

  private maxArg (array: number[]): number {
    let maxIndex = 0
    let maxValue = array[0]
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxIndex = i
        maxValue = array[i]
      }
    }
    return maxIndex
  }

  /**
   * ucbScore - The score for a node is based on its value (Q), plus an exploration bonus based
   * on the prior (P - predicted probability of choosing the action that leads to this node)
   *    Upper Confidence Bound
   *    ```U(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))```
   *    Q(s,a): valueScore - the expected reward for taking action a from state s, i.e. the Q values
   *    N(s,a): the number of times we took action a from state s across simulation
   *    N(s): the sum of all visits at state s across simulation
   *    P(s,a): the initial estimate of taking an action a from the state s according to the
   *    policy returned by the current neural network. Part of the policyScore.
   *    c is a hyperparameter that controls the degree of exploration. The specific choice of c depends on
   *    the problem domain and the trade-off between exploration and exploitation that you want to achieve.
   *    Typically, you might choose c empirically through experimentation or use domain knowledge to guide its selection.
   *    Common values for c are often in the range of 1 to 2, but this can vary based on the problem characteristics and
   *    the desired behavior of the algorithm. If you're unsure about the problem characteristics, or you want to
   *    encourage more initial exploration, you might lean towards larger values of c - like c = 2.
   * The point of the UCB is to initially prefer actions with high prior probability (P) and low visit count (N),
   * but asymptotically prefer actions with a high action value (Q).
   * @param totalVisits N(s)
   * @param child
   * @param minMaxStats
   * @param initQValue mean Q(s,a) for all visited children
   * @private
   */
  private ucbScore (totalVisits: number, child: ChildNode, minMaxStats: Normalizer, initQValue: number): number {
    const valueScore = child.visits > 0 ? minMaxStats.normalize(child.qValue()) : initQValue
    const c = this.config.pbCinit + Math.log((1 + totalVisits + this.config.pbCbase) / this.config.pbCbase)
    const policyScore = c * child.prior * Math.sqrt(totalVisits) / (1 + child.visits)
    return valueScore + policyScore
  }

  // Estimate Q value for non-visited actions
  private calculateInitQValue (parent: Node, minMaxStats: Normalizer): number {
    const qValues = parent.children.filter(child => child.visits > 0).map(child => child.qValue())
    // initialize min/max bounds
    minMaxStats.update(parent.qValue())
    // prepare for normalization
    qValues.forEach(qValue => {
      minMaxStats.update(qValue)
    })
    return minMaxStats.normalize(qValues.reduce((sum, qValue) => sum + qValue, 0) / (qValues.length + 1))
  }

  /** seq_halving.py - Functions for Sequential Halving
   * Returns the action selected by Sequential Halving with Gumbel.
   *
   *   Initially, we sample `max_num_considered_actions` actions without replacement.
   *   From these, the actions with the highest `gumbel + logits + qvalues` are
   *   visited first.
   *
   *      Require: k: number of actions.
   *      Require: m ≤ k: number of actions sampled without replacement.
   *      Require: n: number of simulations.
   *      Require: logits ∈ Rk: predictor logits from a policy network π.
   *        Sample k Gumbel variables:
   *          (g ∈ Rk) ∼ Gumbel(0)
   *        Find m actions with the highest g(a) + logits(a):
   *          Atopm = argtop(g + logits, m)
   *        Use Sequential Halving with n simulations to identify the best action from the Atopm actions,
   *        by comparing g(a) + logits(a) + σ(ˆq(a)).
   *          An+1 = argmax a∈Remaining(g(a) + logits(a) + σ(ˆq(a)))
   *        return An+1
   *
   * @param rootNode the node from which to take an action.
   * @param numSimulations the simulation budget.
   * @param maxNumConsideredActions the number of actions sampled without replacement.
   * @returns action: the action selected from the given node.
   * @protected
   */
  protected gumbelMuZeroRootActionSelection (rootNode: Node, numSimulations: number, maxNumConsideredActions: number): Action {
    // Get the table of considered visits
    const table = this.getTableOfConsideredVisits(maxNumConsideredActions, numSimulations)
    // Get the number of actions to be considered
    const numConsidered = Math.min(maxNumConsideredActions, rootNode.possibleActions.length)
    // At the root, simulationIndex equals the sum of visit counts
    const simulationIndex = rootNode.children.reduce((s, child) => s + child.visits, 0)
    // Fetch considered visit from the table
    const consideredVisit = table[numConsidered][simulationIndex]

    const actionIndex = tf.tidy(() => {
      const priorLogits = rootNode.childrenLogits(this.config.actionSpace)
      const completedQvalues = this.qtransformCompletedByMixValue(rootNode)

      // Calculate the score of the considered actions
      const toArgmax = this.scoreConsidered(
        consideredVisit,
        priorLogits,
        completedQvalues,
        rootNode.childrenVisits(this.config.actionSpace)
      )

      // Mask the invalid actions at the root
      return this.maskedArgmax(toArgmax, rootNode.possibleActions)
    })
    return rootNode.possibleActions[actionIndex]
  }

  /**
   * Selects the action with a deterministic action selection.
   *
   *   The action is selected based on the visit counts to produce visitation
   *   frequencies similar to softmax(prior_logits + qvalues).
   *
   * @param node the node from which to take an action.
   * @returns action: the action selected from the given node.
   * @protected
   */
  protected gumbelMuzeroInteriorActionSelection (
    node: Node
  ): Action {
    const actionIndex = tf.tidy(() => {
      // Extract visit counts and prior logits for the given node.
      const visitCounts = node.childrenVisits(this.config.actionSpace)
      const priorLogits = node.childrenLogits(this.config.actionSpace)
      // Calculate completed Q-values using the provided qtransform function.
      const completedQvalues = this.qtransformCompletedByMixValue(node)

      debug(`visits: ${visitCounts.toString()}`)
      debug(`prior: ${priorLogits.toString()}`)
      debug(`Qvalues: ${completedQvalues.toString()}`)

      // Compute `prior_logits + completed_qvalues` for an improved policy.
      const logitsPlusQvalues: tf.Tensor1D = tf.add(priorLogits, completedQvalues)
      debug(`logitsPlusQvalues: ${logitsPlusQvalues.toString()}`)
      const softmaxValues: tf.Tensor1D = tf.softmax(logitsPlusQvalues)
      debug(`softmaxValues: ${softmaxValues.toString()}`)
      // Prepare the input for argmax selection based on visit counts and softmax values.
      const toArgmax = this.prepareArgmaxInput(softmaxValues, visitCounts)
      // Use argmax to select the action.
      debug(`toArgMax: ${toArgmax.toString()}`)
      return softmaxValues.argMax().bufferSync().get(0)
    })
    return node.children.find(child => child.action.id === actionIndex)?.action ?? node.children[0].action
  }

  /**
   * Prepares the input for the deterministic selection.
   *
   *   When calling argmax(_prepare_argmax_input(...)) multiple times
   *   with updated visit_counts, the produced visitation frequencies will
   *   approximate the probs.
   *
   *   For the derivation, see Section 5 "Planning at non-root nodes" in
   *   "Policy improvement by planning with Gumbel":
   *   https://openreview.net/forum?id=bERaNdoegnO
   *
   * @param probs a policy or an improved policy. Shape `[num_actions]`.
   * @param visitCounts the existing visit counts. Shape `[num_actions]`.
   * @returns The input to an argmax. Shape `[num_actions]`.
   * @private
   */
  private prepareArgmaxInput (probs: tf.Tensor1D, visitCounts: tf.Tensor1D): tf.Tensor {
    // Ensure probs and visitCounts have the same shape.
    if (probs.shape[0] !== visitCounts.shape[0]) {
      throw new Error('probs and visitCounts must have the same shape')
    }

    // Calculate the sum of visit counts and reshape to keep dimensions.
    const sumVisitCounts = tf.sum(visitCounts).reshape([1])
    debug(`sumVisits: ${sumVisitCounts.toString()}`)

    // Compute the argmax input using `probs - visitCounts / (1 + sumVisitCounts)`.
    const adjustedVisitCounts = tf.div(visitCounts, tf.add(1, sumVisitCounts))
    debug(`adjustedVisitCounts: ${adjustedVisitCounts.toString()}`)
    return tf.sub(probs, adjustedVisitCounts)
  }

  /**
   * Returns a valid action with the highest `to_argmax`.
   * @param toArgmax
   * @param possibleActions
   * @private
   */
  private maskedArgmax (toArgmax: tf.Tensor, possibleActions: Action[]): number {
    return tf.tidy(() => {
      const actionMap = tf.oneHot(tf.tensor1d(possibleActions.map(action => action.id), 'int32'), toArgmax.shape[0], 1, 0, 'bool')
      // Set -Infinity for invalid actions
      const maskedValues = tf.where(actionMap, toArgmax, tf.fill(toArgmax.shape, -Infinity))
      // Return index of the maximum valid action
      return maskedValues.argMax(-1).cast('int32').bufferSync().get(0)
    })
  }

  /**
   * Returns completed qvalues.
   *
   *   The missing Q-values of the unvisited actions are replaced by the
   *   mixed value, defined in Appendix D of
   *   "Policy improvement by planning with Gumbel":
   *   https://openreview.net/forum?id=bERaNdoegnO
   *
   *   The Q-values are transformed by a linear transformation:
   *     `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.
   *
   * @param node the parent node.
   * @returns Completed Q-values. Shape `[num_actions]`.
   * @private
   */
  private qtransformCompletedByMixValue (node: Node): tf.Tensor1D {
    // scale for the Q-values.
    const valueScale = 0.1
    // offset to the `max(visit_counts)` in the scaling factor.
    const maxvisitInit = 50.0
    // if True, scale the qvalues by `1 / (max_q - min_q)`.
    const rescaleValues = true
    // if True, complete the Q-values with mixed value, otherwise complete the Q-values with the raw value.
    const useMixedValue = true
    // the minimum denominator when using `rescale_values`.
    const epsilon = 1e-8

    const qvalues = node.qValues(this.config.actionSpace)
    const visitCounts = node.childrenVisits(this.config.actionSpace)

    debug(`visits: ${visitCounts.toString()}`)
    debug(`Qvalues: ${qvalues.toString()}`)

    // Compute mixed value and produce completed Qvalues
    const rawValue = node.rawValue
    const priorProbs = tf.softmax(node.childrenLogits(this.config.actionSpace))
    debug(`priorProps: ${priorProbs.toString()}`)
    const value = useMixedValue ? this.computeMixedValue(rawValue, qvalues, visitCounts, priorProbs) : rawValue
    debug(`value: ${value}`)
    let completedQvalues = this.completeQvalues(qvalues, visitCounts, value)
    debug(`completedQvalues: ${completedQvalues.toString()}`)

    // Scale Q-values if required
    if (rescaleValues) {
      completedQvalues = this.rescaleQvalues(completedQvalues, epsilon)
      debug(`scaled completedQvalues: ${completedQvalues.toString()}`)
    }
    const maxvisit = tf.max(visitCounts)
    debug(`maxvisits: ${maxvisit.toString()}`)
    const visitScale = tf.add(maxvisitInit, maxvisit)
    debug(`visitScale: ${visitScale.toString()}`)
    return tf.mul(tf.mul(visitScale, valueScale), completedQvalues)
  }

  /**
   * Rescales the given completed Q-values to be from the [0, 1] interval.
   * @param qvalues
   * @param epsilon
   * @private
   */
  private rescaleQvalues (qvalues: tf.Tensor1D, epsilon: number): tf.Tensor1D {
    const minValue = tf.min(qvalues)
    const maxValue = tf.max(qvalues)
    return tf.div(tf.sub(qvalues, minValue), tf.maximum(tf.sub(maxValue, minValue), epsilon))
  }

  /**
   * Returns completed Q-values, with the `value` for unvisited actions.
   * @param qvalues
   * @param visitCounts
   * @param value
   * @private
   */
  private completeQvalues (qvalues: tf.Tensor1D, visitCounts: tf.Tensor1D, value: number): tf.Tensor1D {
    // Replace missing Q-values with value for unvisited actions
    return tf.where(tf.greater(visitCounts, 0), qvalues, tf.fill([qvalues.shape[0]], value))
  }

  /**
   * Interpolates the raw_value and weighted qvalues.
   *
   * @param rawValue an approximate value of the state.
   * @param qvalues Q-values for all actions. Shape `[num_actions]`. The unvisited
   *       actions have undefined Q-value.
   * @param visitCounts the visit counts for all actions. Shape `[num_actions]`.
   * @param priorProbs the action probabilities, produced by the policy network for
   *       each action. Shape `[num_actions]`.
   * @returns An estimator of the state value.
   * @private
   */
  private computeMixedValue (rawValue: number, qvalues: tf.Tensor1D, visitCounts: tf.Tensor1D, priorProbs: tf.Tensor1D): number {
    return tf.tidy(() => {
      const sumVisitCounts = tf.sum(visitCounts)
      // Ensuring non-nan weighted_q, even if the visited actions have zero prior probability.
      const probs = tf.maximum(tf.fill(priorProbs.shape, Number.EPSILON), priorProbs)
      // Summing the probabilities of the visited actions.
      const sumProbs = tf.sum(tf.where(tf.greater(visitCounts, 0), probs, 0))
      const weightedQ = tf.sum(
        tf.where(
          tf.greater(visitCounts, 0),
          tf.div(tf.mul(probs, qvalues), tf.where(tf.greater(visitCounts, 0), sumProbs, 1)),
          0
        )
      )
      return tf.div(tf.add(rawValue, tf.mul(sumVisitCounts, weightedQ)), tf.add(sumVisitCounts, 1)).bufferSync().get(0)
    })
  }

  /**
   * Returns a sequence of visit counts considered by Sequential Halving.
   *
   *   Sequential Halving is a "pure exploration" algorithm for bandits, introduced
   *   in "Almost Optimal Exploration in Multi-Armed Bandits":
   *   http://proceedings.mlr.press/v28/karnin13.pdf
   *
   *   The visit counts allows to implement Sequential Halving by selecting the best
   *   action from the actions with the currently considered visit count.
   *
   *      Require: k: number of actions.
   *      Require: m ≤ k: number of actions sampled without replacement.
   *      Require: n: number of simulations.
   *      Require: logits ∈ Rk: predictor logits from a policy network π.
   *        Sample k Gumbel variables:
   *          (g ∈ Rk) ∼ Gumbel(0)
   *        Find m actions with the highest g(a) + logits(a):
   *          Atopm = argtop(g + logits, m)
   *        Use Sequential Halving with n simulations to identify the best action from the Atopm actions,
   *        by comparing g(a) + logits(a) + σ(ˆq(a)).
   *          An+1 = arg maxa∈Remaining (g(a) + logits(a) + σ(ˆq(a)))
   *        return An+1
   *
   * @param maxNumConsideredActions The maximum number of considered actions.
   *      The `maxNumConsideredActions` can be smaller than the number of actions.
   * @param numSimulations The total simulation budget.
   * @returns A array with visit counts. Length `num_simulations`.
   *
   */
  private getSequenceOfConsideredVisits (maxNumConsideredActions: number, numSimulations: number): number[] {
    if (maxNumConsideredActions <= 1) {
      return Array.from({ length: numSimulations }, (_, i) => i)
    }

    const log2max = Math.ceil(Math.log2(maxNumConsideredActions))
    const visits: number[] = new Array(maxNumConsideredActions).fill(0)

    const sequence: number[] = []
    let numConsidered = maxNumConsideredActions
    while (sequence.length < numSimulations) {
      const numExtraVisits = Math.max(1, Math.floor(numSimulations / (log2max * numConsidered)))
      for (let i = 0; i < numExtraVisits; i++) {
        sequence.push(...visits.slice(0, numConsidered))
        for (let j = 0; j < numConsidered; j++) {
          visits[j]++
        }
      }
      // Halve the number of considered actions
      numConsidered = Math.max(2, Math.floor(numConsidered / 2))
    }
    return sequence.slice(0, numSimulations)
  }

  /**
   * Returns a table of sequences of visit counts.
   *
   * @param maxNumConsideredActions The maximum number of considered actions (the number of actions sampled without
   *       replacement). The `maxNumConsideredActions` can be smaller than the number of actions.
   * @param numSimulations The total simulation budget.
   * @returns A tuple of sequences of visit counts.
   *     Shape [max_num_considered_actions + 1, num_simulations].
   *
   */
  protected getTableOfConsideredVisits (maxNumConsideredActions: number, numSimulations: number): number[][] {
    return Array.from({ length: maxNumConsideredActions + 1 }, (_, m) =>
      this.getSequenceOfConsideredVisits(m, numSimulations)
    )
  }

  /**
   * Returns a score usable for an argmax.
   *
   * @param consideredVisit
   * @param logits
   * @param normalizedQvalues
   * @param visitCounts
   */

  private scoreConsidered (
    consideredVisit: number,
    logits: tf.Tensor1D,
    normalizedQvalues: tf.Tensor1D,
    visitCounts: tf.Tensor1D
  ): tf.Tensor {
    const lowLogit = -1e9

    // Normalize logits by subtracting max value for numerical stability
    const maxLogits = logits.max(-1, true)
    logits = logits.sub(maxLogits)

    // Create penalty where only children with considered visits are not penalized
    const penalty = tf.where(
      tf.equal(visitCounts, consideredVisit),
      tf.zerosLike(visitCounts),
      tf.fill(visitCounts.shape, -Infinity)
    )

    const loc = 0.0
    const scale = 1.0
    const gumbel = tf.tidy(() => {
      // Step 1: Generate uniform random numbers in (0, 1)
      const U = tf.randomUniform(logits.shape, 0, 1)
      // Step 2: Apply the Gumbel transformation: loc - scale * log(-log(U))
      const negLogU = tf.neg(tf.log(tf.neg(tf.log(U))))
      return tf.add(tf.mul(negLogU, scale), loc)
    })

    // Ensure shapes match
    tf.util.assertShapesMatch(
      gumbel.shape,
      normalizedQvalues.shape,
      'Shape mismatch between \'gumbel\' and \'normalizedQvalues\''
    )
    tf.util.assertShapesMatch(
      gumbel.shape,
      penalty.shape,
      'Shape mismatch between \'gumbel\' and \'penalty\''
    )

    // Compute final score
    return tf.maximum(tf.fill(gumbel.shape, lowLogit), gumbel.add(logits).add(normalizedQvalues)).add(penalty)
  }
}
