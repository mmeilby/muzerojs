import { SharedStorage } from './muzero/training/sharedstorage'
import { ReplayBuffer } from './muzero/replaybuffer/replaybuffer'
import { SelfPlay } from './muzero/selfplay/selfplay'
import { Training } from './muzero/training/training'
import { Validate } from './muzero/validation/validate'
import type { Environment } from './muzero/games/core/environment'
import { type Config } from './muzero/games/core/config'
import fs from 'fs'
import debugFactory from 'debug'
import * as tf from '@tensorflow/tfjs-node-gpu'

const output = debugFactory('muzero:muzero:output')
// Enable general output like titles, warnings, and errors (along with environmental DEBUG settings)
debugFactory.enable('muzero:muzero:output,'.concat(process.env.DEBUG ?? ''))

export class Muzero {
  public trainingSession (factory: Environment, conf: Config): void {
    output(`Running training session for MuZero acting on ${conf.savedNetworkPath} environment`)
    output(`Tensorflow backend: ${tf.getBackend()}`)
    output(`Backends: ${tf.engine().backendNames().join(',')}`)
    let lastStep = 0
    try {
      const json = fs.readFileSync(`data/${conf.savedNetworkPath}/muzero.json`, { encoding: 'utf8' })
      if (json !== null) {
        lastStep = JSON.parse(json) as number
      }
    } catch (e) {
      // Error: ENOENT: no such file or directory, open 'data/path/muzero.json'
      if ((e as Error).message.includes('ENOENT')) {
        output('Configuration file does not exist. A new file will be created.')
      } else {
        output(e)
      }
    }
    const sharedStorage = new SharedStorage(conf)
    const replayBuffer = new ReplayBuffer(conf)
    replayBuffer.loadSavedGames(factory)
    const selfPlay = new SelfPlay(conf, factory)
    const train = new Training(conf, lastStep)
    const validate = new Validate(conf, factory)
    sharedStorage.loadNetwork().then(() => {
      Promise.all([
        selfPlay.runSelfPlay(sharedStorage, replayBuffer),
        train.trainNetwork(sharedStorage, replayBuffer),
        validate.logMeasures(sharedStorage, lastStep)
      ]).then(() => {
        output(`--- Performance: ${replayBuffer.performance().toFixed(1)}%`)
        output(`--- Accuracy (${sharedStorage.networkCount}): ${train.statistics().toFixed(2)}`)
        lastStep += sharedStorage.networkCount
        fs.writeFileSync(`data/${conf.savedNetworkPath}/muzero.json`, JSON.stringify(lastStep), 'utf8')
        output(`Training completed - training sessions completed: ${lastStep}`)
      }).catch((err) => {
        throw err
      })
    }).catch((err) => {
      output(err)
    })
  }

  public testSession (factory: Environment, conf: Config): void {
    const sharedStorage = new SharedStorage(conf)
    sharedStorage.loadNetwork().then(() => {
      const network = sharedStorage.latestNetwork()
      const validate = new Validate(conf, factory)
      const outcomes: number[] = [0, 0, 0]
      output('Running 1000 test sessions')
      for (let i = 0; i < 1000; i++) {
        const aiPlayer = i % 2 === 0 ? 1 : -1
        const result = validate.battle(network, aiPlayer) + 1
        outcomes[result] = outcomes[result] + 1
      }
      output('Battle AI wins', outcomes[2])
      output('Battle AI ties', outcomes[1])
      output('Battle AI losses', outcomes[0])
    }).catch((err) => {
      output(err)
    })
  }

  public printModels (conf: Config): void {
    output(`Nuværende backend: ${tf.getBackend()}`)
    output(`Backends: ${tf.engine().backendNames().join(',')}`)
    if (conf.modelGenerator !== undefined) {
      const model = conf.modelGenerator()
      model.print()
    }
  }
}
