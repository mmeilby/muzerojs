import * as tf from '@tensorflow/tfjs-node-gpu'
import { describe, expect, test } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { type MuZeroNimState } from '../../muzero/games/nim/nimstate'
import { type Action } from '../../muzero/games/core/action'
import { MuZeroNimAction } from '../../muzero/games/nim/nimaction'

describe('Nim Unit Test:', () => {
  const factory = new MuZeroNim()
  test('Check the Nim Game', async () => {
    const state = factory.reset()
    // validate move labels
    expect(new MuZeroNimAction().set('H?-?').id).toEqual(-1)
    expect(new MuZeroNimAction().set('H2-2').toString()).toEqual('H2-2')
    expect(new MuZeroNimAction().set('H5-1').toString()).toEqual('H5-1')
    expect(state.board.toString()).toEqual('1,2,3,4,5')
    expect(factory.toString(state)).toEqual('1|2|3|4|5')
    expect(factory.legalActions(state).map(a => a.toString())).toEqual(
      ['H1-1', 'H2-1', 'H2-2', 'H3-1', 'H3-2', 'H3-3', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(state.observation.equal(tf.tensor4d([[
      [[1], [0], [0], [0], [0]],
      [[1], [1], [0], [0], [0]],
      [[1], [1], [1], [0], [0]],
      [[1], [1], [1], [1], [0]],
      [[1], [1], [1], [1], [1]]]]))).toBeTruthy()
    const s1 = factory.step(state, new MuZeroNimAction().set('H1-1'))
    expect(s1.board.toString()).toEqual('0,2,3,4,5')
    expect(factory.legalActions(s1).map(a => a.toString())).toEqual(
      ['H2-1', 'H2-2', 'H3-1', 'H3-2', 'H3-3', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    const s2 = factory.step(s1, new MuZeroNimAction().set('H3-3'))
    expect(s2.board.toString()).toEqual('0,2,0,4,5')
    expect(factory.legalActions(s2).map(a => a.toString())).toEqual(
      ['H2-1', 'H2-2', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.expertAction(s2).toString()).toEqual('H2-1')
    tf.tidy(() => {
      const pi = factory.expertActionPolicy(s2)
      expect(pi.sum().bufferSync().get(0)).toBeGreaterThan(0.9999)
      expect(pi.argMax().bufferSync().get(0)).toEqual(1)
    })
    const s3 = factory.step(s2, new MuZeroNimAction().set('H2-1'))
    expect(s3.board.toString()).toEqual('0,1,0,4,5')
    expect(factory.toString(s3)).toEqual('_|1|_|4|5')
    expect(factory.legalActions(s3).map(a => a.toString())).toEqual(
      ['H2-1', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s3)).toEqual(false)
    const s4 = factory.step(s3, new MuZeroNimAction().set('H2-1'))
    expect(s4.board.toString()).toEqual('0,0,0,4,5')
    expect(factory.toString(s4)).toEqual('_|_|_|4|5')
    expect(factory.legalActions(s4).map(a => a.toString())).toEqual(
      ['H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s4)).toEqual(false)
    expect(factory.reward(s4, s4.player)).toEqual(0.1)
    const s5 = factory.step(s4, new MuZeroNimAction().set('H4-4'))
    expect(s5.board.toString()).toEqual('0,0,0,0,5')
    expect(factory.toString(s5)).toEqual('_|_|_|_|5')
    expect(factory.legalActions(s5).map(a => a.toString())).toEqual(
      ['H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s5)).toEqual(false)
    expect(factory.reward(s5, s5.player)).toEqual(0.1)
    const s6 = factory.step(s5, new MuZeroNimAction().set('H5-4'))
    expect(s6.board.toString()).toEqual('0,0,0,0,1')
    expect(factory.toString(s6)).toEqual('_|_|_|_|1')
    expect(factory.legalActions(s6).length).toEqual(0)
    expect(factory.terminal(s6)).toBeTruthy()
    expect(factory.reward(s6, s6.player)).toEqual(-1)
    const s7 = factory.step(s6, new MuZeroNimAction().set('H5-1'))
    expect(s7.board.toString()).toEqual('0,0,0,0,0')
    expect(factory.legalActions(s7).length).toEqual(0)
    expect(factory.expertAction(s7).id).toEqual(-1)
    tf.tidy(() => {
      const pi = factory.expertActionPolicy(s7)
      expect(pi.sum().bufferSync().get(0)).toEqual(0)
      expect(pi.max().bufferSync().get(0)).toEqual(0)
    })
    expect(factory.terminal(s7)).toBeTruthy()
    expect(s7.player).toEqual(-1)
    expect(factory.reward(s7, s7.player)).toEqual(1)
    expect(factory.reward(s7, -s7.player)).toEqual(-1)
    const [player, board, history] = JSON.parse(factory.serialize(s4))
    expect(player).toEqual(1)
    expect(board).toEqual([0, 0, 0, 4, 5])
    expect(history.map((a: number) => new MuZeroNimAction(a).toString())).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
    const deserializedState = factory.deserialize(factory.serialize(s4))
    expect(deserializedState.player).toEqual(1)
    expect(deserializedState.board).toEqual([0, 0, 0, 4, 5])
    expect(deserializedState.history.map((a: Action) => a.toString())).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
  })
  test('Check specific Nim Game', async () => {
    const state = factory.reset()
    expect(state.board.toString()).toEqual('1,2,3,4,5')
    const s1 = factory.step(state, new MuZeroNimAction().set('H3-1'))
    expect(s1.board.toString()).toEqual('1,2,2,4,5')
    const s2 = factory.step(s1, new MuZeroNimAction().set('H5-2'))
    expect(s2.board.toString()).toEqual('1,2,2,4,3')
    const s3 = factory.step(s2, new MuZeroNimAction().set('H4-2'))
    expect(s3.board.toString()).toEqual('1,2,2,2,3')
    const s4 = factory.step(s3, new MuZeroNimAction().set('H1-1'))
    expect(s4.board.toString()).toEqual('0,2,2,2,3')
    const s5 = factory.step(s4, new MuZeroNimAction().set('H5-1'))
    expect(s5.board.toString()).toEqual('0,2,2,2,2')
    const s6 = factory.step(s5, new MuZeroNimAction().set('H2-1'))
    expect(s6.board.toString()).toEqual('0,1,2,2,2')
    expect('H3-1,H4-1,H5-1'.includes(factory.expertAction(s6).toString())).toBeTruthy()
    tf.tidy(() => {
      const pi = factory.expertActionPolicy(s6)
      expect(pi.sum().bufferSync().get(0)).toBeGreaterThan(0.9999)
      expect(pi.max().bufferSync().get(0)).toEqual(0.2823789119720459)
      expect(pi.min().bufferSync().get(0)).toEqual(0)
    })
    const s7 = factory.step(s6, new MuZeroNimAction().set('H3-1'))
    expect(s7.board.toString()).toEqual('0,1,1,2,2')
  })
  test('Check reward system', () => {
    const simulate = (moves: string): MuZeroNimState => {
      let state = factory.reset()
      for (const move of moves.split(':')) {
        state = factory.step(state, new MuZeroNimAction().set(move))
      }
      return state
    }
    const preset = (board: string): MuZeroNimState => {
      const state = factory.reset()
      board.split('|').forEach((stack, heap) => {
        state.board[heap] = stack.localeCompare('_') === 0 ? 0 : Number.parseInt(stack)
      })
      return state
    }
    const s1 = simulate('H2-1:H5-2:H4-4:H1-1:H2-1:H3-3:H5-2')
    expect(factory.reward(s1, 1)).toEqual(1)
    const s2 = simulate('H4-4:H5-5:H3-3:H1-1:H2-2')
    expect(factory.reward(s2, -1)).toEqual(1)
    const rewardTest: Array<[string, number]> = [
      ['1|2|3|4|5', 0.1],
      ['1|2|3|4|4', -0.1],
      ['1|1|1|4|4', 0.1],
      ['1|1|1|4|5', -0.1],
      ['1|1|1|1|1', -0.1],
      ['1|1|1|1|_', 0.1],
      ['1|_|_|_|_', -1],
      ['_|_|_|4|5', 0.1],
      ['_|_|_|_|5', 0.1],
      ['_|_|_|_|_', 1]
    ]
    rewardTest.forEach(([state, reward]) => {
      const b = preset(state)
      expect(factory.reward(b, 1)).toEqual(reward)
      expect(factory.reward(b, -1)).toEqual(-reward)
    })
  })
})
