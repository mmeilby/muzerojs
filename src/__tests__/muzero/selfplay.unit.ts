import { describe, test, expect } from '@jest/globals'
import { MuZeroSharedStorage } from '../../muzero/training/sharedstorage'
import { MuZeroAction } from '../../muzero/games/core/action'
import { MuZeroReplayBuffer } from '../../muzero/replaybuffer/replaybuffer'
import { MuZeroSelfPlay } from '../../muzero/selfplay/selfplay'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { type MuZeroNimState } from '../../muzero/games/nim/nimstate'
import { NimNetModel } from '../../muzero/games/nim/nimmodel'
import { MuZeroConfig } from '../../muzero/games/core/config'
import { type MuZeroNetObservation } from '../../muzero/networks/network'
import { MuZeroMockedNetwork } from '../../muzero/networks/mnetwork'
import * as tf from '@tensorflow/tfjs-node'
import { MuZeroNimUtil } from '../../muzero/games/nim/nimutil'

describe('Muzero Self Play Unit Test:', () => {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const support = new MuZeroNimUtil()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  test('Check stopGradient', () => {
    // Define placeholders
    const x = tf.variable(tf.ones([3, 2]))
    const y = tf.variable(tf.zeros([3, 4]))

    // Define variables
    const w1 = tf.variable(tf.ones([2, 3]))
    const w2 = tf.variable(tf.ones([3, 4]))

    const scaleGradiant = (tensor: tf.Tensor, scale: number): tf.Tensor => {
      return tf.tidy(() => {
        const tidyTensor = tf.variable(tensor, false)
        return tensor.mul(scale).add(tidyTensor.mul(1 - scale))
      })
    }
    // Define the computation graph
    const loss = (): tf.Scalar => {
      const hidden = x.matMul(w1)
      const output = hidden.matMul(w2)
      const e = output.sub(y)
      return scaleGradiant(e, 0.5).norm().asScalar()
    }
    const optimizer = tf.train.sgd(1)

    // Run the TensorFlow.js session
    console.log('*****before gradient descent*****')
    console.log('w1---\n', w1.arraySync(), '\n', 'w2---\n', w2.arraySync())

    //      x.assign(tf.randomNormal([3, 2]));
    //      y.assign(tf.randomNormal([3, 4]));

    optimizer.minimize(loss)

    console.log('*****after gradient descent*****')
    console.log('w1---\n', w1.arraySync(), '\n', 'w2---\n', w2.arraySync())
  })
  test('Check self play', async () => {
    const replayBuffer = new MuZeroReplayBuffer<MuZeroNimState, MuZeroAction>(conf)
    const sharedStorage = new MuZeroSharedStorage<MuZeroAction>(conf)
    const network = new MuZeroMockedNetwork<MuZeroNimState, MuZeroAction>(factory)
    await sharedStorage.saveNetwork(1, network)
    conf.selfPlaySteps = 2
    const selfPlay = new MuZeroSelfPlay(conf, factory, model)
    await selfPlay.runSelfPlay(sharedStorage, replayBuffer)
    expect(replayBuffer.numPlayedGames).toEqual(2)
  })
  test('Check reward system', () => {
    const simulate = (moves: string): MuZeroNimState => {
      let state = factory.reset()
      for (const move of moves.split(':')) {
        state = factory.step(state, support.actionFromString(move))
      }
      return state
    }
    const s1 = simulate('H2-1:H5-2:H4-4:H1-1:H2-1:H3-3:H5-2')
    expect(factory.reward(s1, 1)).toEqual(1)
    const s2 = simulate('H4-4:H5-5:H3-3:H1-1:H2-2')
    expect(factory.reward(s2, -1)).toEqual(1)
  })
})
