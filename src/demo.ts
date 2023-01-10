import { MuZeroSharedStorage } from './muzero/training/sharedstorage'
import { MuZeroNim } from './muzero/games/nim/nim'
import { NimNetModel } from './muzero/games/nim/nimmodel'
import { MuZeroAction } from './muzero/games/core/action'
import debugFactory from 'debug'

const debug = debugFactory('muzero:demo:play ')

async function run (): Promise<void> {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const sharedStorage = new MuZeroSharedStorage({
    observationSize: model.observationSize,
    actionSpaceSize: config.actionSpaceSize
  })
  const network = await sharedStorage.latestNetwork()
  let state = factory.reset()
  const currentObservation = model.observation(state)
  let networkOutput = network.initialInference(currentObservation)
  while (!factory.terminal(state)) {
    let bestAction: number | undefined
    const bestPolicy = 0
    networkOutput.policyMap.forEach((score, id) => {
      if (score >= bestPolicy) {
        bestAction = id
      }
    })
    if (bestAction !== undefined) {
      debug(`--- Best action: ${bestAction} ${state.toString()}`)
      state = factory.step(state, new MuZeroAction(bestAction))
      networkOutput = network.recurrentInference(networkOutput.hiddenState, networkOutput.policy)
    } else {
      break
    }
  }
  debug(`--- Done ${state.toString()}`)
}

run().then(res => {}).catch(err => console.error(err))
