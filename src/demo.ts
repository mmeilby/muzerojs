import { SharedStorage } from './muzero/training/sharedstorage'
import { MuZeroNim } from './muzero/games/nim/nim'
import debugFactory from 'debug'
import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Action } from './muzero/selfplay/mctsnode'

const debug = debugFactory('muzero:demo:play ')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const conf = factory.config()
  const sharedStorage = new SharedStorage(conf)
  const network = sharedStorage.latestNetwork()
  let state = factory.reset()
  const currentObservation = state.observation
  let networkOutput = network.initialInference(currentObservation.expandDims(0))
  while (!factory.terminal(state)) {
    // select the most popular action
    const bestAction = tf.multinomial(networkOutput.tfPolicy as tf.Tensor1D, 1, undefined, false) as tf.Tensor1D
    const legalActions = factory.legalActions(state)
    const action: Action = { id: bestAction.bufferSync().get(0) }
    if (legalActions.find(a => a.id === action.id) != null) {
      state = factory.step(state, action)
      debug(`--- Best action: ${action.id} ${state.toString()}`)
      const tfAction = tf.oneHot(tf.tensor1d([action.id], 'int32'), conf.actionSpace, 1, 0, 'float32')
      networkOutput = network.recurrentInference(networkOutput.tfHiddenState, tfAction)
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
      const tfAction = tf.oneHot(tf.tensor1d([nextBestAction.id], 'int32'), conf.actionSpace, 1, 0, 'float32')
      networkOutput = network.recurrentInference(networkOutput.tfHiddenState, tfAction)
    }
  }
  debug(`--- Done ${state.toString()}`)
}

run().then(_ => {
}).catch(err => {
  console.error(err)
})
