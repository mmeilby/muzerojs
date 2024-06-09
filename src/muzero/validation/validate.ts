import { GameHistory } from '../selfplay/gamehistory'
import * as tf from '@tensorflow/tfjs-node-gpu'
import debugFactory from 'debug'
import { type Network } from '../networks/nnet'
import { type Action } from '../games/core/action'
import { NetworkState } from '../networks/networkstate'
import type { Config } from '../games/core/config'
import type { Environment } from '../games/core/environment'
import type { SummaryFileWriter } from '@tensorflow/tfjs-node-gpu/dist/tensorboard'
import { type SharedStorage } from '../training/sharedstorage'

const perf = debugFactory('muzero:validate:perf')

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

  /**
   *
   * @param sharedStorage
   * @param lastStep
   */
  public async logMeasures (sharedStorage: SharedStorage, lastStep: number = 0): Promise<void> {
    while (sharedStorage.networkCount < this.config.trainingSteps) {
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
   * Measure invalid outcome for policy prediction.
   * Invalid outcome prediction is predicted probability for actions not allowed for a specific state.
   * Invalid actions should be predicted with no probability (0%). This validation measures the mean deviation.
   * @param network The network to use for the measurement
   * @returns number The mean probability for invalid outcome over a number of random games.
   * Only game states with possible actions are included.
   */
  public testPolicyInvalidOutcomePrediction (network: Network): number {
    return tf.tidy(() => {
      const invalidOutcomes: number[] = []
      const gameHistory = new GameHistory(this.env, this.config)
      // Play a game from start to end, register target data on the fly for the game history
      while (!gameHistory.terminal() && gameHistory.historyLength() < this.config.maxMoves) {
        const networkOutput = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
        const policy = networkOutput.tfPolicy.squeeze().arraySync() as number[]
        const legalPolicy: number[] = gameHistory.legalPolicy(policy)
        const iop = legalPolicy.reduce((s, v, i) => v === 0 ? s + policy[i] : s, 0) * 100
        perf(`--- Test state: ${gameHistory.state.toString()}`)
        perf(`--- Test policy: ${legalPolicy.map(v => v.toFixed(2)).toString()}`)
        perf(`--- Policy invalid outcome prediction: ${iop.toFixed(1)}%`)
        const a = this.randomChoice(legalPolicy)
        if (a >= 0) {
          invalidOutcomes.push(iop)
          const action = this.env.actionRange()[a]
          perf(`--- Test action: ${a} ${action?.toString() ?? ''}`)
          gameHistory.apply(action)
        }
      }
      perf(`--- Invalid outcome prediction: ${invalidOutcomes.map(v => v.toFixed(2)).toString()}`)
      return invalidOutcomes.reduce((s, v) => s + v, 0) / invalidOutcomes.length
    })
  }

  /**
   * Validate the network ability to predict the same hidden state for initial and recurrent inference.
   * The validation measures the absolute difference between the predicted hidden state from the h(o) and g(s,a) networks.
   * The validation uses one simulated game based on random weighted choices for each move.
   * At each state the initial inference predicts the hidden state based on the tracked game true state.
   * The recurrent inference predicts the hidden state based on the previous predicted state and the chosen action.
   * At the game start only the hidden state is predicted using the initial inference.
   * @param network The network to use for the validation
   * @returns number A mean value for the absolute difference of the hidden states for all the game moves.
   * A mean value of 0 indicates a similar prediction for initial and recurrent inference
   */
  public testHiddenStates (network: Network): number {
    return tf.tidy(() => {
      let action: Action | undefined
      let hiddenState: tf.Tensor | undefined
      let policy: number[]
      const deviations: tf.Tensor[] = []
      const gameHistory = new GameHistory(this.env, this.config)
      // Play a game from start to end, register target data on the fly for the game history
      while (!gameHistory.terminal() && gameHistory.historyLength() < this.config.maxMoves) {
        if (hiddenState !== undefined && action !== undefined) {
          const networkOutputRef = network.recurrentInference(new NetworkState(hiddenState, [gameHistory.state]), [action])
          hiddenState = networkOutputRef.tfHiddenState
          policy = networkOutputRef.tfPolicy.squeeze().arraySync() as number[]
          gameHistory.apply(action)
          const networkOutput = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
          const deviation = tf.losses.absoluteDifference(networkOutput.tfHiddenState, networkOutputRef.tfHiddenState)
          deviations.push(deviation)
        } else {
          const networkOutput = network.initialInference(new NetworkState(gameHistory.makeImage(-1), [gameHistory.state]))
          hiddenState = networkOutput.tfHiddenState
          policy = networkOutput.tfPolicy.squeeze().arraySync() as number[]
        }
        const legalPolicy: number[] = gameHistory.legalPolicy(policy)
        const outcast = legalPolicy.reduce((s, v, i) => v === 0 ? s + policy[i] : s, 0) * 100
        perf(`--- Test state: ${gameHistory.state.toString()}`)
        perf(`--- Test policy: ${legalPolicy.map(v => v.toFixed(2)).toString()}`)
        perf(`--- Policy false actions: ${outcast.toFixed(1)}%`)
        const a = this.randomChoice(legalPolicy)
        action = this.env.actionRange()[a]
        perf(`--- Test action: ${a} ${action?.toString() ?? ''}`)
      }
      perf(`--- Hidden states absolute difference: ${deviations.toString()}`)
      return tf.concat(deviations).mean().bufferSync().get(0)
    })
  }
}
