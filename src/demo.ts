import { SharedStorage } from './muzero/training/sharedstorage'
import { MuZeroNim } from './muzero/games/nim/nim'
import debugFactory from 'debug'
import * as tf from '@tensorflow/tfjs-node-gpu'

import { type Action } from './muzero/games/core/action'
import { MuZeroNimAction } from './muzero/games/nim/nimaction'
import { NetworkState } from './muzero/networks/networkstate'

const debug = debugFactory('muzero:demo:play ')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  const sharedStorage = new SharedStorage(conf)
  const network = sharedStorage.latestNetwork()
  let state = factory.reset()
  const currentObservation = state.observation.expandDims(0)
  let networkOutput = network.initialInference(new NetworkState(currentObservation))
  while (!factory.terminal(state)) {
    // select the most popular action
    const bestAction = tf.multinomial(networkOutput.tfPolicy as tf.Tensor1D, 1, undefined, false) as tf.Tensor1D
    const legalActions = factory.legalActions(state)
    const action: Action = new MuZeroNimAction(bestAction.bufferSync().get(0))
    if (legalActions.find(a => a.id === action.id) != null) {
      state = factory.step(state, action)
      debug(`--- Best action: ${action.id} ${state.toString()}`)
      networkOutput = network.recurrentInference(new NetworkState(networkOutput.tfHiddenState), [action])
    } else {
      let nextBestAction = action
      let nextBestPolicy = 0
      const policyMap = networkOutput.tfPolicy.arraySync() as number[]
      legalActions.forEach(a => {
        if (policyMap[a.id] > nextBestPolicy) {
          nextBestAction = a
          nextBestPolicy = policyMap[a.id]
        }
      })
      state = factory.step(state, nextBestAction)
      debug(`--- Next best (legal) action: ${nextBestAction.id} ${state.toString()} - ${action.id} was invalid - policy: [${policyMap.map(v => v.toFixed(4)).toString()}]`)
      networkOutput = network.recurrentInference(new NetworkState(networkOutput.tfHiddenState), [nextBestAction])
    }
  }
  debug(`--- Done ${state.toString()}`)
}

run().then(_ => {
}).catch(err => {
  console.error(err)
})
