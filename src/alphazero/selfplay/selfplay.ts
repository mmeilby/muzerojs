import { GameHistory } from './gamehistory'
import { ObservationModel } from '../games/core/model'
import { Environment } from '../games/core/environment'
import { TranspositionTable, DataGateway } from './data-store'
import { Config } from '../games/core/config'
import { Network } from '../networks/nnet'
import { Actionwise } from '../games/core/actionwise'
import { Statewise } from '../games/core/statewise'
import { MCTSNode } from './mctsnode'
import { MCTSState } from './mctsstate'
import debugFactory from 'debug'

/* eslint @typescript-eslint/no-var-requires: "off" */
const { jStat } = require('jstat')
const debug = debugFactory('alphazero:selfplay:module')
const debugNodes = debugFactory('alphazero:selfplay:debug')

/**
 * MuZeroSelfPlay - where the games are played
 */
export class SelfPlay<State extends Statewise, Action extends Actionwise> {
  constructor (
    private readonly config: Config,
    private readonly env: Environment<State, Action>,
    private readonly model: ObservationModel<State>,
    // The network to predict the rewards and policies for each state
    private readonly network: Network<Action>
  ) {
  }

  /**
   * executeEpisode - play a full game using the network to decide the moves
   * This function executes one episode of self-play, starting with player 1.
   * Each game is produced by starting at the initial board position, then
   * repeatedly executing a Monte Carlo Tree Search to generate moves until the end
   * of the game is reached.
   *
   * It uses a temp=1 if move < tempThreshold, and thereafter uses temp=0.
   *
   * @returns Game history for a single played game containing state, pi, and value.
   *          pi is the MCTS informed policy vector, value is +1 if
   *          the player eventually won the game, else -1.
   */
  public executeEpisode (): GameHistory<State, Action> {
    const dataStore = new TranspositionTable<State>(new Map())
    const gameHistory = new GameHistory(this.env)
    const rootNode: MCTSNode<State, Action> = new MCTSNode(this.mctsState(gameHistory.state, dataStore), gameHistory.state.player)
    // Play a game from start to end, register target data on the fly for the game history
    for (let node = rootNode; !gameHistory.terminal() && gameHistory.recordedSteps() < this.config.maxMoves;) {
      const temp = gameHistory.episodeStep() < this.config.temperatureThreshold ? 1 : 0
      const pi = this.getActionProp(node, dataStore, temp)
      const action = this.env.action(this.randomChoice(pi))
      gameHistory.apply(action)
      gameHistory.storeSearchStatistics(pi)
      // Prepare new root node for tree search
      const bestChild = node.children.find(child => child.action.id === action.id)
      if (bestChild !== undefined) {
        node = bestChild
      } else {
        throw new Error(`Unexpected missing child node should represent best action from state ${this.env.toString(gameHistory.state)}`)
      }
    }
    gameHistory.updateRewards()
    this.debugNode(rootNode)
    return gameHistory
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

  public predictAction (state: State): Action {
    const dataStore = new TranspositionTable<State>(new Map())
    const rootNode: MCTSNode<State, Action> = new MCTSNode(this.mctsState(state, dataStore), state.player)
    const pi = this.getActionProp(rootNode, dataStore, 0)
    // Find the best probability
    const max = pi.reduce((m, v) => Math.max(m, v))
    // Make a list of the best actions
    const bestActions = pi.map((p, i) => { return { p, i } }).filter(p => p.p === max).map(p => p.i)
    // Pick one randomly (if we can't assume the void action - the pi array may be empty)
    const bestAction = bestActions.at(Math.floor(Math.random() * bestActions.length)) ?? -1
    return this.env.action(bestAction)
  }

  /**
   * getActionProp - Core Monte Carlo Tree Search algorithm
   * To decide on an action, we run N simulations, always starting at the root of
   * the search tree and traversing the tree according to the UCB formula until we
   * reach a leaf node.
   *
   * Returns: a policy vector where the probability of the ith action is
   *          proportional to Nsa[(s,a)]**(1./temp)
   *
   * @param rootNode The node to start the tree search from
   * @param dataStore The data store to keep track of the visit counts and discounted rewards
   * @param temp The temperature used for policy determination
   */
  private getActionProp (
    rootNode: MCTSNode<State, Action>,
    dataStore: DataGateway<State>,
    temp: number): number[] {
    for (let sim = 0; sim < this.config.simulations; sim++) {
      const v = this.search(rootNode, dataStore)
      rootNode.Qsa = (rootNode.Nsa * rootNode.Qsa + v) / (rootNode.Nsa + 1)
      rootNode.Nsa++
      if (!rootNode.isRootNode) {
        rootNode.Ns++
      }
    }
    return rootNode.policy(this.config.actionSpace, temp)
  }

  /**
   * search
   * This function performs one iteration of MCTS. It is recursively called
   * till a leaf node is found. The action chosen at each node is one that
   * has the maximum upper confidence bound as in the paper.
   *
   * Once a leaf node is found, the neural network is called to return an
   * initial policy P and a value v for the state. This value is propagated
   * up the search path. In case the leaf node is a terminal state, the
   * outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
   * updated.
   *
   * NOTE: the return values are the negative of the value of the current
   * state. This is done since v is in [-1,1] and if v is the value of a
   * state for the current player, then its value is -v for the other player.
   *
   * Returns: the negative of the value of the current canonicalBoard
   *
   * @param node
   * @param dataStore
   */
  protected search (
    node: MCTSNode<State, Action>,
    dataStore: DataGateway<State>
  ): number {
    if (node.isNewNode()) {
      node.Es = this.env.reward(node.mctsState.state, node.player)
    }
    // Return reward if this node is the terminal state
    if (node.Es !== 0) {
      return -node.Es
    }
    if (!node.isExpanded()) {
      // We use the representation function to
      // obtain a policy and a reward value given the current observation.
      const currentObservation = this.model.observation(node.mctsState.state)
      const networkOutput = this.network.initialInference(currentObservation)
      this.expandNode(node, networkOutput.policyMap, dataStore)
      debug(`Network predicted reward value: ${JSON.stringify(networkOutput.nValue)}`)
      // Return network predicted reward value
      return -networkOutput.nValue
    }
    const bestChild = this.selectChild(node)
    const v = this.search(bestChild, dataStore)
    bestChild.Qsa = (bestChild.Nsa * bestChild.Qsa + v) / (bestChild.Nsa + 1)
    bestChild.Nsa++
    bestChild.Ns++
    return -v
  }

  /**
   * selectChild - Select the child node with the highest UCB score
   * @param node
   * @private
   */
  private selectChild (node: MCTSNode<State, Action>): MCTSNode<State, Action> {
    const ucbMap = node.children.map(child => {
      return { child, ucb: this.ucbScore(child) }
    }).sort((u1, u2) => u2.ucb - u1.ucb)
    debug(`UCB map: node: ${node.mctsState.state.toString()} ${ucbMap.map(ucb => `${this.env.actionToString(ucb.child.action.id)} => ${ucb.ucb}`)}`)
    return ucbMap[0].child
  }

  /**
   * ucbScore - The score for a node is based on its value, plus an exploration bonus based
   * on the prior (predicted probability of choosing the action that leads to this node)
   *    Upper Confidence Bound
   *    U(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   *    Q(s,a): the expected reward for taking action a from state s, i.e. the Q values
   *    N(s,a): the number of times we took action a from state s across simulation
   *    N(s): the sum of all visits at state s across simulation
   *    P(s,a): the initial estimate of taking an action a from the state s according to the
   *    policy returned by the current neural network.
   *    cpuct is a hyperparameter that controls the degree of exploration
   * The point of the UCB is to initially prefer actions with high prior probability (P) and low visit count (N),
   * but asymptotically prefer actions with a high action value (Q).
   * @param child
   * @param exploit
   * @private
   */
  private ucbScore (child: MCTSNode<State, Action>, exploit = false): number {
    if (exploit) {
      // For exploitation only we simply measure the number of visits
      // Pseudo code actually calculates softmax(child.visits): exp(child.Ns / temp) / sum(exp(child.Ns / temp))
      return child.Nsa
    }
    const pbCbase = this.config.pbCbase
    const pbCinit = this.config.pbCinit // = 1.25
    const cpuct = 1 // Math.log((child.Ns + pbCbase + 1) / pbCbase) + pbCinit
    return child.Qsa + cpuct * child.Psa * Math.sqrt(child.Ns) / (1 + child.Nsa)
  }

  /**
   * expandNode - We expand a node using the value, reward and policy prediction obtained from
   * the neural network.
   * @param node
   * @param policyMap
   * @param dataStore
   * @private
   */
  private expandNode (node: MCTSNode<State, Action>, policyMap: number[], dataStore: DataGateway<State>): void {
    // Mask possible actions and create a policy map inferred by the network
    const policy = Array<number>(this.config.actionSpace).fill(0)
    const possibleActions = this.env.legalActions(node.mctsState.state)
    possibleActions.forEach(action => policy[action.id] = policyMap[action.id])
    debug(`Network policy prediction: ${JSON.stringify(policy)}`)

    if (policy.every(p => p === 0)) {
      // if all valid moves were masked make all valid moves equally probable

      // NB! All valid moves may be masked if either your NNet architecture is insufficient
      // or you've got overfitting or something else.
      // If you have got dozens or hundreds of these messages you should pay attention
      // to your NNet and/or training process.
      debug(`All valid moves were masked, doing a workaround. Network policy: ${JSON.stringify(policyMap)}`)
      possibleActions.forEach(action => policy[action.id] = 1 / possibleActions.length)
    }

    const policySum = policy.reduce((s, p) => s + p)
    possibleActions.forEach(action => {
      const state = this.env.step(node.mctsState.state, action)
      const mctsState = this.mctsState(state, dataStore)
      const child = node.addChild(mctsState, action, state.player)
      // Save the predicted prior probability of choosing the action that leads to this node - P(s,a)
      child.Psa = policy[action.id] / policySum
    })
  }

  /**
   * createMCTSState
   * @param state
   * @param dataStore
   * @private
   */
  private mctsState (state: State, dataStore: DataGateway<State>): MCTSState<State> {
    // Check to see if state is already in DataStore
    let mctsState = dataStore.get(state)
    // If it isn't, create a new MCTSState and store it
    if (mctsState == null) {
      mctsState = new MCTSState(state)
      dataStore.set(state, mctsState)
    }
    return mctsState
  }

  // At the start of each search, we add dirichlet noise to the prior of the root
  // to encourage the search to explore new actions.
  // For chess, root_dirichlet_alpha = 0.3
  private addExplorationNoise (node: MCTSNode<State, Action>): void {
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
      child.Psa = child.Psa * (1 - frac) + noise[i] * frac
    })
  }

  private debugNode (node: MCTSNode<State, Action>, index = 0): void {
    debugNodes(`${'.'.repeat(index)} ${this.env.toString(node.mctsState.state)}: a=${node.isRootNode ? '-' : this.env.actionToString(node.action.id)} Ns=${node.isRootNode ? 0 : node.Ns} Nsa=${node.Nsa} Psa=${node.Psa} Qsa=${node.Qsa} Es=${node.isNewNode() ? '-' : node.Es}`)
    node.children.forEach(child => this.debugNode(child, index + 1))
  }
}
