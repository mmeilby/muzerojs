import * as tf from '@tensorflow/tfjs-node-gpu'
import { describe, expect, test } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { MockedNetwork } from '../../muzero/networks/implementations/mocked'
import { GameHistory } from '../../muzero/selfplay/gamehistory'
import debugFactory from 'debug'
import { MuZeroNimAction } from '../../muzero/games/nim/nimaction'
import { NetworkState } from '../../muzero/networks/networkstate'
import { CoreNet } from '../../muzero/networks/implementations/core'
import { MlpNet } from '../../muzero/networks/implementations/mlp'

const debug = debugFactory('muzero:unit:debug')

describe('Network Unit Test:', () => {
  const factory = new MuZeroNim()
  const config = factory.config()
  config.modelGenerator = () => new MlpNet(config.observationSize, config.actionSpace) // TODO: Is this really needed?
  const mockedNetwork = new MockedNetwork(factory)
  const network = new CoreNet(config)
  test('Check mocked initial inference', () => {
    const gameHistory = new GameHistory(factory, config)
    const hiddenState = tf.stack([gameHistory.makeImage(-1), gameHistory.makeImage(-1), gameHistory.makeImage(-1)])
    const states = [gameHistory.state, gameHistory.state, gameHistory.state]
    const no = mockedNetwork.initialInference(new NetworkState(hiddenState, states))
    expect(JSON.stringify(states)).toEqual(JSON.stringify(no.state))
  })
  test('Check mocked recurrent inference', () => {
    const gameHistory = new GameHistory(factory, config)
    const hiddenState = tf.stack([gameHistory.makeImage(-1), gameHistory.makeImage(-1), gameHistory.makeImage(-1)])
    const states = [gameHistory.state, gameHistory.state, gameHistory.state]
    const no = mockedNetwork.initialInference(new NetworkState(hiddenState, states))
    const action = new MuZeroNimAction(0)
    const nor = mockedNetwork.recurrentInference(new NetworkState(no.tfHiddenState, no.state), [action, action, action])
    expect(nor.state?.toString()).toEqual('0 | H1-1 | 0-2-3-4-5,0 | H1-1 | 0-2-3-4-5,0 | H1-1 | 0-2-3-4-5')
  })
  test('Check recurrent inference', () => {
    const gameHistory = new GameHistory(factory, config)
    const initialState = gameHistory.makeImage(-1)
    const hiddenState = tf.stack([initialState, initialState, initialState])
    const no = network.initialInference(new NetworkState(hiddenState))
    const action = new MuZeroNimAction(0)
    const nor = network.recurrentInference(new NetworkState(no.tfHiddenState), [action, action, action])
    expect(nor.tfHiddenState.shape).toEqual([3, ...config.observationSize])
  })
})
