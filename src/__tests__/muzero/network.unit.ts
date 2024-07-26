import * as tf from '@tensorflow/tfjs-node-gpu'
import { describe, expect, test } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { MockedNetwork } from '../../muzero/networks/implementations/mocked'
import { GameHistory } from '../../muzero/selfplay/gamehistory'
import debugFactory from 'debug'
import { MuZeroNimAction } from '../../muzero/games/nim/nimaction'
import { NetworkState } from '../../muzero/networks/networkstate'
import { CoreNet } from '../../muzero/networks/implementations/core'
import { Validate } from '../../muzero/validation/validate'
import { NimNet } from '../../muzero/networks/implementations/nim'
import { ResNet } from '../../muzero/networks/implementations/conv'

const debug = debugFactory('muzero:unit:debug')

describe('Network Unit Test:', () => {
  const factory = new MuZeroNim()
  const config = factory.config()
  const mockedNetwork = new MockedNetwork(factory)
  test('Check mocked initial inference', () => {
    const gameHistory = new GameHistory(factory, config)
    const hiddenState = tf.concat([gameHistory.makeImage(-1), gameHistory.makeImage(-1), gameHistory.makeImage(-1)])
    const states = [gameHistory.state, gameHistory.state, gameHistory.state]
    const no = mockedNetwork.initialInference(new NetworkState(hiddenState, states))
    expect(JSON.stringify(states)).toEqual(JSON.stringify(no.state))
  })
  test('Check mocked recurrent inference', () => {
    const gameHistory = new GameHistory(factory, config)
    const hiddenState = tf.concat([gameHistory.makeImage(-1), gameHistory.makeImage(-1), gameHistory.makeImage(-1)])
    const states = [gameHistory.state, gameHistory.state, gameHistory.state]
    const no = mockedNetwork.initialInference(new NetworkState(hiddenState, states))
    const action = new MuZeroNimAction(0)
    const nor = mockedNetwork.recurrentInference(new NetworkState(no.tfHiddenState, no.state), [action, action, action])
    expect(nor.state?.toString()).toEqual('0 | H1-1 | 0-2-3-4-5,0 | H1-1 | 0-2-3-4-5,0 | H1-1 | 0-2-3-4-5')
  })
  test('Check mocked policy invalid outcome', () => {
    const validate = new Validate(config, factory)
    let deviation = 0
    for (let i = 0; i < 10; i++) {
      deviation += validate.testPolicyInvalidOutcomePrediction(mockedNetwork)
    }
    expect(deviation).toEqual(0)
  })
  test('Check mocked hidden state predictions', async () => {
    const validate = new Validate(config, factory)
    let deviation = 0
    for (let i = 0; i < 10; i++) {
      deviation += validate.testHiddenStates(mockedNetwork)
    }
    expect(deviation).toEqual(0)
  })
  test('Check mocked random battle', () => {
    const validate = new Validate(config, factory)
    let deviation = 0
    for (let i = 0; i < 100; i++) {
      deviation += validate.battle(mockedNetwork, i % 2 === 0 ? 1 : -1)
    }
    expect(deviation).toBeGreaterThan(49)
  })
  test('Check recurrent inference state shape for NimNet', () => {
    config.modelGenerator = () => new NimNet(config)
    const network = new CoreNet(config)
    const gameHistory = new GameHistory(factory, config)
    const initialState = gameHistory.makeImage(-1)
    const hiddenState = tf.concat([initialState, initialState, initialState])
    const no = network.initialInference(new NetworkState(hiddenState))
    const action = new MuZeroNimAction(0)
    const nor = network.recurrentInference(new NetworkState(no.tfHiddenState), [action, action, action])
    expect(nor.tfHiddenState.shape).toEqual([3, ...config.observationSize])
  })
  test('Check recurrent inference state shape for ResNet', () => {
    config.modelGenerator = () => new ResNet(config.observationSize, config.actionSpace, config.observationSize, config.actionShape)
    const network = new CoreNet(config)
    const gameHistory = new GameHistory(factory, config)
    const initialState = gameHistory.makeImage(-1)
    const hiddenState = tf.concat([initialState, initialState, initialState])
    const no = network.initialInference(new NetworkState(hiddenState))
    const action = new MuZeroNimAction(0)
    const nor = network.recurrentInference(new NetworkState(no.tfHiddenState), [action, action, action])
    expect(nor.tfHiddenState.shape).toEqual([3, ...config.observationSize])
  })
})
