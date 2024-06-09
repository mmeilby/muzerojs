import { describe, test } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { SharedStorage } from '../../muzero/training/sharedstorage'
import { Validate } from '../../muzero/validation/validate'
import debugFactory from 'debug'

const perf = debugFactory('muzero:validate:test')
debugFactory.enable('muzero:validate:test')

describe('Nim network performance Test:', () => {
  const factory = new MuZeroNim()
  const config = factory.config()
  config.savedNetworkPath = config.savedNetworkPath.concat('test')
  const sharedStorage = new SharedStorage(config)
  test('Measure hidden state deviation', async () => {
    await sharedStorage.loadNetwork()
    const validate = new Validate(config, factory)
    let deviation = 0
    for (let i = 0; i < 10; i++) {
      deviation += validate.testHiddenStates(sharedStorage.latestNetwork())
    }
    perf(`Mean hidden state deviation: ${(deviation / 10).toFixed(3)}`)
  })
  test('Validate invalid outcome for policy prediction', async () => {
    await sharedStorage.loadNetwork()
    const validate = new Validate(config, factory)
    let deviation = 0
    for (let i = 0; i < 10; i++) {
      deviation += validate.testPolicyInvalidOutcomePrediction(sharedStorage.latestNetwork())
    }
    perf(`Mean policy deviation: ${(deviation / 10).toFixed(2)}%`)
  })
})
