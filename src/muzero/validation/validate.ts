import * as tf from '@tensorflow/tfjs-node-gpu'
import debugFactory from 'debug'
import { type Network } from '../networks/nnet'
import { type Action } from '../games/core/action'
import { NetworkState } from '../networks/networkstate'
import type { Config } from '../games/core/config'
import type { Environment } from '../games/core/environment'
import type { SummaryFileWriter } from '@tensorflow/tfjs-node-gpu/dist/tensorboard'
import { type SharedStorage } from '../training/sharedstorage'
import { type State } from '../games/core/state'

const perf = debugFactory('muzero:validate:perf')
const output = debugFactory('muzero:muzero:output')

export class Validate {
  private readonly summary: SummaryFileWriter

  constructor (
    private readonly config: Config,
    private readonly env: Environment,
    private readonly logDir = `./logs/muzero/${config.savedNetworkPath}`
  ) {
    this.summary = tf.node.summaryFileWriter(this.logDir)
  }

  /**
   *
   * @param sharedStorage
   * @param lastStep
   */
  public async logMeasures (sharedStorage: SharedStorage, lastStep: number = 0): Promise<void> {
    while (sharedStorage.networkCount < this.config.trainingSteps) {
      const it = sharedStorage.networkCount > 0 ? sharedStorage.networkCount / this.config.checkpointInterval : 0
      const total = Math.floor(this.config.trainingSteps / this.config.checkpointInterval)
      output(`Got new model - version ${it} of ${total}`)
      const locale = 'dk-DK'
      const memoryUsage = process.memoryUsage()
      output(`Total use (RSS): ${(memoryUsage.rss / 1024 / 1024).toLocaleString(locale, { maximumFractionDigits: 0 })} Mb`)
      output(`Heap Total (V8): ${(memoryUsage.heapTotal / 1024 / 1024).toLocaleString(locale, { maximumFractionDigits: 0 })} Mb`)
      output(`Heap Used (V8): ${(memoryUsage.heapUsed / 1024 / 1024).toLocaleString(locale, { maximumFractionDigits: 0 })} Mb`)
      output(`External (C++ objects): ${(memoryUsage.external / 1024 / 1024).toLocaleString(locale, { maximumFractionDigits: 0 })} Mb`)
      output(`Tensor use: ${tf.memory().numTensors}`)
      // Measure hidden state deviation
      let deviation = 0
      for (let i = 0; i < 10; i++) {
        deviation += this.testHiddenStates(sharedStorage.latestNetwork())
      }
      this.summary.scalar('hidden_state_deviation', deviation / 10, sharedStorage.networkCount + lastStep)
      // Validate invalid outcome for policy prediction
      let iop = 0
      for (let i = 0; i < 10; i++) {
        iop += this.testPolicyInvalidOutcomePrediction(sharedStorage.latestNetwork())
      }
      this.summary.scalar('policy_invalid_outcome', iop / 10, sharedStorage.networkCount + lastStep)
      // Test drive the network against random agents
      const outcomes: number[] = [0, 0, 0]
      for (let i = 0; i < 100; i++) {
        const aiPlayer = i % 2 === 0 ? 1 : -1
        const result = this.battle(sharedStorage.latestNetwork(), aiPlayer) + 1
        outcomes[result] = outcomes[result] + 1
      }
      this.summary.scalar('battle_ai_wins', outcomes[2], sharedStorage.networkCount + lastStep)
      this.summary.scalar('battle_ai_loss', outcomes[0], sharedStorage.networkCount + lastStep)
      // Wait for the trained network version based on the game plays just added to the replay buffer
      await sharedStorage.waitForUpdate()
    }
  }

  /**
   * randomChoice - make a weighted random choice based on the policy applied
   * @param policy Policy containing a normalized probability vector
   * @returns The action index of the policy with the most probability randomly chosen. If policy is empty (no valid actions) then -1 is returned.
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
    return i < policy.length ? i : -1
  }

  /**
   * <h3>Measure Invalid Outcome for Policy Prediction</h3>
   * This function measures the network's ability to predict only valid actions for a given state.
   * An invalid outcome prediction occurs when the network assigns a non-zero probability to an action
   * that is not allowed for a specific state. The goal is for the network to assign a probability of 0%
   * to all disallowed actions.
   *
   * Invalid outcome prediction is quantified as the mean probability of predicting any disallowed action across multiple game simulations. A higher mean probability indicates that the network requires more training to correctly predict valid actions. Only game states with at least one valid action are considered in this measurement.
   * @param network The network to use for the measurement
   * @returns number: The mean probability of predicting invalid outcomes over a number of random game states.
   * <ul>
   * <li>A mean probability of 0 indicates perfect prediction, where no invalid actions are predicted.</li>
   * <li>A higher mean probability suggests that the network often predicts disallowed actions,
   * indicating a need for further training.</li>
   * </ul>
   */
  public testPolicyInvalidOutcomePrediction (network: Network): number {
    return tf.tidy(() => {
      const invalidOutcomeProbabilities: number[] = []
      const actionRange = this.env.actionRange()
      let state = this.env.reset()
      // Play a game from start to end, register target data on the fly for the game history
      while (!this.env.terminal(state)) {
        const networkOutput = network.initialInference(new NetworkState(state.observation, [state]))
        const policy = networkOutput.policy
        // Map/mask the policy to legal moves according to the environment
        const legalPolicy: number[] = this.legalPolicy(state, policy)
        // Calculate the total probability for an invalid outcome (the probabilities for invalid outcome are identified previously)
        const iop = legalPolicy.reduce((s, v, i) => v === 0 ? s + policy[i] : s, 0) * 100
        perf(`--- Test state: ${state.toString()}`)
        perf(`--- Test policy: ${legalPolicy.map(v => v.toFixed(2)).toString()}`)
        perf(`--- Policy invalid outcome probability: ${iop.toFixed(1)}%`)
        const a = this.randomChoice(legalPolicy)
        if (a >= 0) {
          invalidOutcomeProbabilities.push(iop)
          const action = actionRange[a]
          perf(`--- Test action: ${a} ${action?.toString() ?? ''}`)
          state = this.env.step(state, action)
        }
      }
      perf(`--- Invalid outcome probabilities: ${invalidOutcomeProbabilities.map(v => v.toFixed(2)).toString()}`)
      return invalidOutcomeProbabilities.reduce((s, v) => s + v, 0) / invalidOutcomeProbabilities.length
    })
  }

  /**
   * <h3>Validate Network Consistency for Predicting Hidden States</h3>
   * The purpose of this validation is to ensure that the network can consistently predict the same hidden state
   * for both initial and recurrent inferences.
   * This is crucial for the reliability of the network in predicting future states based on its learned patterns.
   *
   * The validation process measures the absolute difference between the hidden states predicted by two different parts of the network: the initial inference network (h(o)) and the recurrent inference network (g(s,a)). It involves simulating a game where moves are determined by random weighted choices.
   * <ul>
   * <li>Initial Inference: At each game state, the initial inference network predicts the hidden state
   * based on the true state of the game.</li>
   * <li>Recurrent Inference: The recurrent inference network predicts the hidden state based on
   * the previously predicted state and the chosen action.</li>
   * </ul>
   * At the start of the game, the hidden state is predicted only using the initial inference.
   *
   * @param network The network to use for the validation
   * @returns number: The mean absolute difference of the hidden states for all the moves in the game.
   * A mean value of 0 indicates that the initial and recurrent inferences predict similar hidden states,
   * suggesting that the network is consistent.
   * A higher mean value indicates discrepancies between initial and recurrent inferences,
   * implying that the network requires more training to achieve consistent hidden state predictions.
   */
  public testHiddenStates (network: Network): number {
    return tf.tidy(() => {
      let action: Action | undefined
      let hiddenState: tf.Tensor | undefined
      let policy: number[]
      const deviations: tf.Tensor[] = []
      const actionRange = this.env.actionRange()
      let state = this.env.reset()
      // Play a game from start to end, register target data on the fly for the game history
      while (!this.env.terminal(state)) {
        if (hiddenState !== undefined && action !== undefined) {
          const networkOutputRef = network.recurrentInference(new NetworkState(hiddenState, [state]), [action])
          hiddenState = networkOutputRef.tfHiddenState
          policy = networkOutputRef.policy
          state = this.env.step(state, action)
          const networkOutput = network.initialInference(new NetworkState(state.observation, [state]))
          const deviation = tf.losses.absoluteDifference(networkOutput.tfHiddenState, networkOutputRef.tfHiddenState)
          deviations.push(deviation)
        } else {
          const networkOutput = network.initialInference(new NetworkState(state.observation, [state]))
          hiddenState = networkOutput.tfHiddenState
          policy = networkOutput.policy
        }
        // Map/mask the policy to legal moves according to the environment
        const legalPolicy: number[] = this.legalPolicy(state, policy)
        perf(`--- Test state: ${state.toString()}`)
        const a = this.randomChoice(legalPolicy)
        action = actionRange[a]
        perf(`--- Test action: ${a} ${action?.toString() ?? ''}`)
      }
      perf(`--- Hidden states absolute difference: ${deviations.toString()}`)
      return tf.concat(deviations).mean().bufferSync().get(0)
    })
  }

  /**
   * Battle against random agent
   * @param network
   * @param aiPlayer
   */
  public battle (network: Network, aiPlayer: number): number {
    const actionRange = this.env.actionRange()
    let state = this.env.reset()
    // Play a game from start to end, register target data on the fly for the game history
    while (!this.env.terminal(state)) {
      perf(`--- Battle state: ${state.toString()}`)
      if (state.player === aiPlayer) {
        tf.tidy(() => {
          const networkOutput = network.initialInference(new NetworkState(state.observation, [state]))
          // Map/mask the policy to legal moves according to the environment
          const legalPolicy: number[] = this.legalPolicy(state, networkOutput.policy)
          const a = this.randomChoice(legalPolicy)
          const action = actionRange[a]
          perf(`--- AI action: ${a} ${action?.toString() ?? ''}`)
          state = this.env.step(state, action)
        })
      } else {
        const action = actionRange[Math.floor(actionRange.length * Math.random())]
        perf(`--- Agent action: ${action?.toString() ?? ''}`)
        state = this.env.step(state, action)
      }
    }
    return this.env.reward(state, aiPlayer)
  }

  /**
   * Return a policy with probabilities only for allowed actions.
   * Illegal actions are assigned the probability of 0.
   * The probabilities are adjusted so the sum is 1.0 even though some actions are masked away.
   * @param state
   * @param policy
   */
  private legalPolicy (state: State, policy: number[]): number[] {
    const legalActions = this.env.legalActions(state).map(a => a.id)
    const legalPolicy: number[] = policy.map((v, i) => legalActions.includes(i) ? v : 0)
    const psum = legalPolicy.reduce((s, v) => s + v, 0)
    return psum > 0 ? legalPolicy.map(v => v / psum) : legalPolicy
  }

  /**
   * preloadReplayBuffer - fill the replay buffer with new current game plays
   * based on the latest version of the trained network
   * @param replayBuffer
   *
   public buildTestHistory (replayBuffer: ReplayBuffer): void {
   this.unfoldGame(this.env.reset(), [], replayBuffer)
   //    info(`Build history completed - generated ${replayBuffer.numPlayedGames} games with ${replayBuffer.totalSamples} samples`)
   }

   private unfoldGame (state: State, actionList: Action[], replayBuffer: ReplayBuffer): void {
   const actions = this.env.legalActions(state)
   for (const action of actions) {
   const newState = this.env.step(state, action)
   if (this.env.terminal(newState)) {
   const gameHistory = new GameHistory(this.env, this.config)
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
   */
}
