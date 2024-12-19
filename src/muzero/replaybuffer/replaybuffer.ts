import { GameHistory } from '../selfplay/gamehistory'
import { Batch } from './batch'
import fs from 'fs'
import debugFactory from 'debug'
import { type Environment } from '../games/core/environment'
import { type Config } from '../games/core/config'
import * as tf from '@tensorflow/tfjs-node-gpu'

const debug = debugFactory('muzero:replaybuffer:module')

/**
 * Replay Buffer
 */

export class ReplayBuffer {
  public numPlayedGames: number
  public numPlayedSteps: number
  public totalSamples: number
  private buffer: GameHistory[]
  private readonly path: string

  /**
   *
   * @param config
   * @param config.replayBufferSize Number of self-play games to keep in the replay buffer
   * @param config.actionSpace Number of all possible actions
   * @param config.tdSteps Number of steps in the future to take into account for calculating
   * the target value
   * @param config.batchSize Number of parts of games to train on at each training step
   * @param config.numUnrollSteps Number of game moves to keep for every batch element
   * @param config.stackedObservations Number of previous observations and previous actions
   * to add to the current observation
   * @param config.discount Chronological discount of the reward
   * @param config.prioritizedReplay Prioritized Replay (See paper appendix Training),
   * select in priority the elements in the replay buffer which are unexpected for the network
   * @param config.priorityAlpha How much prioritization is used, 0 corresponding to the uniform case,
   * paper suggests 1.0
   */
  public constructor (
    private readonly config: Config
  ) {
    this.buffer = []
    this.numPlayedGames = 0
    this.numPlayedSteps = 0
    this.totalSamples = 0
    this.path = `data/${this.config.savedNetworkPath}/`
    // Create path if needed
    fs.stat(this.path, (err, _stats) => {
      if (err != null) {
        fs.mkdir(this.path, () => {
          debug(`Created game data path: ${this.path}`)
        })
      }
    })
  }

  get totalGames (): number {
    return this.buffer.length
  }

  get lastGame (): GameHistory | undefined {
    return this.buffer.at(-1)
  }

  get games (): GameHistory[] {
    return this.buffer
  }

  /**
   * saveGame
   * Save a game in the replay buffer (generated by selfPlay)
   * @param gameHistory
   */
  public saveGame (gameHistory: GameHistory): void {
    this.setGamePriority(gameHistory)
    if (this.buffer.length >= this.config.replayBufferSize) {
      const delGameHistory = this.buffer.shift()
      if (delGameHistory != null) {
        this.totalSamples -= delGameHistory.rootValues.length
        // Tidy the observation tensors
        delGameHistory.dispose()
      }
    }
    this.buffer.push(gameHistory)
    this.numPlayedGames++
    this.numPlayedSteps += gameHistory.rootValues.length
    this.totalSamples += gameHistory.rootValues.length
    if (this.numPlayedGames % this.config.checkpointInterval === 0) {
      this.storeSavedGames()
    }
  }

  public setGamePriority (gameHistory: GameHistory): void {
    if (this.config.prioritizedReplay) {
      // Initial priorities for the prioritized replay (See paper appendix Training)
      // For each game position calculate the absolute deviation from target value. Largest deviation has priority
      const priorities: number[] = []
      gameHistory.rootValues.forEach((rootValue, i) => {
        const targetValue = gameHistory.computeTargetValue(i, this.config.tdSteps)
        const priority = Math.pow(Math.abs(rootValue - targetValue), this.config.priorityAlpha)
        priorities.push(priority)
      })
      // Deviations are saved for later priority sorting
      gameHistory.priorities = priorities
      gameHistory.gamePriority = gameHistory.priorities.reduce((m, p) => Math.max(m, p), 0)
    }
  }

  /**
   * Get a sample batch from the replay buffer (used for training), one batch for each game selected
   * The total number of batches returned is controlled by config.batchSize
   * This method supports abstract reinforcement learning (MuZero).
   * @param numUnrollSteps Number of game moves to keep for every batch element
   * @param tdSteps Number of steps in the future to take into account for calculating the target value
   * @return Batch[] Sample batch - list of batch elements (observation, action history, and corresponding targets)
   */

  /* Pseudocode
    def sample_batch(self, num_unroll_steps: int, td_steps: int):
      games = [self.sample_game() for _ in range(self.batch_size)]
      game_pos = [(g, self.sample_position(g)) for g in games]
      return [(g.make_image(i), g.history[i:i + num_unroll_steps], g.make_target(i, num_unroll_steps, td_steps, g.to_play())) for (g, i) in game_pos]
   */
  public sampleBatch (numUnrollSteps: number, tdSteps: number): Batch[] {
    return this.sampleGame().map(index => {
      const g = this.buffer[index]
      const position = this.samplePosition(g)
      const actionHistory = g.actionHistory.slice(position, position + numUnrollSteps)
      const target = g.makeTarget(position, numUnrollSteps, tdSteps)
      return new Batch(g.makeImage(position), actionHistory, target)
    })
  }

  public loadSavedGames (
    environment: Environment
  ): void {
    try {
      const json = fs.readFileSync(this.path.concat('games.json'), { encoding: 'utf8' })
      if (json !== null) {
        this.buffer = new GameHistory(environment, this.config).deserialize(json)
        this.totalSamples = this.buffer.reduce((sum, game) => sum + game.rootValues.length, 0)
        this.numPlayedGames = this.buffer.length
        this.numPlayedSteps = this.totalSamples
      }
    } catch (e) {
      // Error: ENOENT: no such file or directory, open 'data/path/games.json'
      if ((e as Error).message.includes('ENOENT')) {
        debug('Saved games file does not exist. A new file will be created.')
      } else {
        debug(e)
      }
    }
  }

  public storeSavedGames (): void {
    const stream = JSON.stringify(this.buffer.map(gh => gh.serialize()))
    fs.writeFileSync(this.path.concat('games.json'), stream, 'utf8')
  }

  /**
   * Estimate replay buffer performance
   * The return value is the percentage of most desired game play from trained network perspective
   * iterated over all game plays in the replay buffer.
   * The more distinct the policy responses are the higher percentage should be expected as the self play
   * engine randomly selects the best move weighted by the policies
   */
  public performance (): number {
    const performanceTotal = this.buffer.reduce((s, game) => s + game.validateEndState(), 0)
    return performanceTotal / this.buffer.length * 100
  }

  public redundance (): number {
    const redundanceMap = new Map<string, number>()
    this.buffer.forEach((g: GameHistory): void => {
      const key = g.actionHistory.join(':')
      const count = redundanceMap.get(key)
      if (count !== undefined) {
        redundanceMap.set(key, count + 1)
      } else {
        redundanceMap.set(key, 1)
      }
    })
    return tf.tidy(() => {
      const dist = tf.tensor1d([...redundanceMap.values()])
      return dist.mean().bufferSync().get(0)
    })
  }

  /**
   * sampleGame - Sample game from buffer either uniformly or according to some priority.
   * See paper appendix Training.
   * @private
   */
  private sampleGame (): number[] {
    if (this.buffer.length > 1) {
      return tf.tidy(() => {
        const gameProbs = tf.tensor1d(this.buffer.map(gameHistory => gameHistory.gamePriority)).log()
        // Define the probability for each game based on popularity (game priorities).
        // Select the most popular games - note that for some reason we need to ask for
        // one more sample as the first one always is fixed
        return tf.multinomial(gameProbs, this.config.batchSize + 1).arraySync().slice(1) as number[]
      })
    } else {
      return this.buffer.map(_ => 0)
    }
  }

  /**
   * samplePosition - Sample position from game either uniformly or according to some priority
   * @param gameHistory
   * @private
   */
  private samplePosition (gameHistory: GameHistory): number {
    return tf.tidy(() => {
      // define the probability for each game position based on priorities (discounted target deviations)
      const probs = tf.tensor1d(gameHistory.priorities).log()
      // select the most popular position - note that for some reason we need to ask for
      // two samples as the first one always is fixed
      return tf.multinomial(probs, 2).bufferSync().get(1)
    })
  }
}
