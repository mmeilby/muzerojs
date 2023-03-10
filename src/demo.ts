import { MuZeroSharedStorage } from './muzero/training/sharedstorage'
import { MuZeroNim } from './muzero/games/nim/nim'
import { NimNetModel } from './muzero/games/nim/nimmodel'
import { MuZeroAction } from './muzero/games/core/action'
import debugFactory from 'debug'
import * as tf from '@tensorflow/tfjs-node'
import { MuZeroConfig } from './muzero/games/core/config'

const debug = debugFactory('muzero:demo:play ')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  const sharedStorage = new MuZeroSharedStorage(conf)
  const network = await sharedStorage.latestNetwork()
  let state = factory.reset()
  const currentObservation = model.observation(state)
  let networkOutput = network.initialInference(currentObservation)
  while (!factory.terminal(state)) {
    // select the most popular action
    const bestAction = tf.multinomial(networkOutput.policyMap, 1, undefined, false) as tf.Tensor1D
    const legalActions = factory.legalActions(state)
    const action = new MuZeroAction(bestAction.bufferSync().get(0))
    if (legalActions.find(a => a.id === action.id) != null) {
      state = factory.step(state, action)
      debug(`--- Best action: ${action.id} ${state.toString()}`)
      networkOutput = network.recurrentInference(networkOutput.aHiddenState, action)
    } else {
      let nextBestAction = action
      let nextBestPolicy = 0
      legalActions.forEach(a => {
        if (networkOutput.policyMap[a.id] > nextBestPolicy) {
          nextBestAction = a
          nextBestPolicy = networkOutput.policyMap[a.id]
        }
      })
      state = factory.step(state, nextBestAction)
      debug(`--- Next best (legal) action: ${nextBestAction.id} ${state.toString()} - ${action.id} was invalid - policy: [${networkOutput.policyMap.map(v => v.toFixed(4)).toString()}]`)
      networkOutput = network.recurrentInference(networkOutput.aHiddenState, nextBestAction)
    }
  }
  debug(`--- Done ${state.toString()}`)
}

run().then(_ => {}).catch(err => console.error(err))
