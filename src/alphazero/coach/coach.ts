import { Config } from '../games/core/config'
import { Environment } from '../games/core/environment'
import { ObservationModel } from '../games/core/model'
import { Statewise } from '../games/core/statewise'
import { Actionwise } from '../games/core/actionwise'
import { ReplayBuffer } from '../replaybuffer/replaybuffer'
import { SelfPlay } from '../selfplay/selfplay'
import { Network } from '../networks/nnet'
import * as tf from '@tensorflow/tfjs-node'

import debugFactory from 'debug'
import { Arena } from './arena'
const debug = debugFactory('alphazero:coach:module')

export class Coach<State extends Statewise, Action extends Actionwise> {
  private readonly skipFirstSelfPlay: boolean

  constructor (
    private readonly config: Config,
    private readonly env: Environment<State, Action>,
    private readonly model: ObservationModel<State>
  ) {
    this.skipFirstSelfPlay = false
  }

  /**
     * learn
     * Performs numIters iterations with numEps episodes of self-play in each
     * iteration. After every iteration, it retrains neural network with
     * examples in trainExamples (which has a maximum length of maxlenofQueue).
     * It then pits the new neural network against the old one and accepts it
     * only if it wins >= updateThreshold fraction of games.
     */
  public async learn (network: Network<Action>): Promise<void> {
    const replayBuffer = new ReplayBuffer<State, Action>(this.config)
    for (let i = 1; i < this.config.numIterations; i++) {
      debug(`Starting coached iteration #${i}`)
      if (!this.skipFirstSelfPlay || i > 1) {
        debug(`Generating ${this.config.numEpisodes} games for training`)
        for (let episodes = 0; episodes < this.config.numEpisodes; episodes++) {
          const mcts = new SelfPlay(this.config, this.env, this.model, network)
          const game = mcts.executeEpisode()
          replayBuffer.saveGame(game)
        }
      }
      // backup history to a file
      // NB! the examples were collected using the model from the previous iteration, so (i-1)
      replayBuffer.storeSavedGames()
      // training new network, keeping a copy of the old one
      const pnet: Network<Action> = Object.create(network)
      network.copyWeights(pnet)
      debug('Training network')
      await this.train(network, replayBuffer)
      // Pitting against previous version
      debug('Pitting against previous model')
      const pmcts = new SelfPlay(this.config, this.env, this.model, pnet)
      const nmcts = new SelfPlay(this.config, this.env, this.model, network)
      const arena = new Arena(this.config, this.env, this.model, nmcts, pmcts)
      const result = arena.playGames(this.config.numGames)
      const nwins = result.oneWon; const pwins = result.twoWon; const draws = result.draws
      // Accept or reject the new model
      debug(`NEW/PREV WINS : ${nwins} / ${pwins} ; DRAWS : ${draws}`)
      if (pwins + nwins === 0 || nwins / (pwins + nwins) < this.config.networkUpdateThreshold) {
        debug('REJECTING NEW MODEL')
        pnet.copyWeights(network)
      } else {
        debug('ACCEPTING NEW MODEL')
        //                network.save('file://data/')
      }
    }
  }

  private async train (network: Network<Action>, replayBuffer: ReplayBuffer<State, Action>): Promise<void> {
    let useBaseline = tf.memory().numTensors
    for (let step = 1; step <= this.config.trainingSteps; step++) {
      const batchSamples = replayBuffer.sampleBatch(this.config.numUnrollSteps, this.config.tdSteps)
      const losses = await network.trainInference(batchSamples)
      debug(`Mean loss: step #${step} ${losses.toFixed(3)}`)
      if (tf.memory().numTensors - useBaseline > 0) {
        debug(`TENSOR USAGE IS GROWING: ${tf.memory().numTensors - useBaseline}`)
        useBaseline = tf.memory().numTensors
      }
    }
  }
}
