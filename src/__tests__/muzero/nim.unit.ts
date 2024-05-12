import * as tf from '@tensorflow/tfjs-node-gpu'
import { describe, expect, test } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { MuZeroNimUtil } from '../../muzero/games/nim/nimutil'
import { type Action } from '../../muzero/selfplay/mctsnode'
import { MuZeroNimState } from '../../muzero/games/nim/nimstate'

describe('Nim Unit Test:', () => {
  const factory = new MuZeroNim()
  const support = new MuZeroNimUtil()
  test('Check the Nim Game', async () => {
    const state = factory.reset()
    // validate move labels
    expect(support.actionFromString('H?-?').id).toEqual(-1)
    expect(support.actionToString(support.actionFromString('H2-2'))).toEqual('H2-2')
    expect(support.actionToString(support.actionFromString('H5-1'))).toEqual('H5-1')
    expect(state.board.toString()).toEqual('1,2,3,4,5')
    expect(factory.toString(state)).toEqual('1|2|3|4|5')
    expect(factory.legalActions(state).map(a => support.actionToString(a))).toEqual(
      ['H1-1', 'H2-1', 'H2-2', 'H3-1', 'H3-2', 'H3-3', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(state.observation.equal(tf.tensor3d([
      [[1], [0], [0], [0], [0]],
      [[1], [1], [0], [0], [0]],
      [[1], [1], [1], [0], [0]],
      [[1], [1], [1], [1], [0]],
      [[1], [1], [1], [1], [1]]]))).toBeTruthy()
    const s1 = factory.step(state, support.actionFromString('H1-1'))
    expect(s1.board.toString()).toEqual('0,2,3,4,5')
    expect(factory.legalActions(s1).map(a => support.actionToString(a))).toEqual(
      ['H2-1', 'H2-2', 'H3-1', 'H3-2', 'H3-3', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    const s2 = factory.step(s1, support.actionFromString('H3-3'))
    expect(s2.board.toString()).toEqual('0,2,0,4,5')
    expect(factory.legalActions(s2).map(a => support.actionToString(a))).toEqual(
      ['H2-1', 'H2-2', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(support.actionToString(factory.expertAction(s2))).toEqual('H2-1')
    tf.tidy(() => {
      const pi = factory.expertActionPolicy(s2)
      expect(pi.sum().bufferSync().get(0)).toBeGreaterThan(0.9999)
      expect(pi.argMax().bufferSync().get(0)).toEqual(1)
    })
    const s3 = factory.step(s2, support.actionFromString('H2-1'))
    expect(s3.board.toString()).toEqual('0,1,0,4,5')
    expect(factory.toString(s3)).toEqual('_|1|_|4|5')
    expect(factory.legalActions(s3).map(a => support.actionToString(a))).toEqual(
      ['H2-1', 'H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s3)).toEqual(false)
    const s4 = factory.step(s3, support.actionFromString('H2-1'))
    expect(s4.board.toString()).toEqual('0,0,0,4,5')
    expect(factory.toString(s4)).toEqual('_|_|_|4|5')
    expect(factory.legalActions(s4).map(a => support.actionToString(a))).toEqual(
      ['H4-1', 'H4-2', 'H4-3', 'H4-4', 'H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s4)).toEqual(false)
    expect(factory.reward(s4, s4.player)).toEqual(0)
    const s5 = factory.step(s4, support.actionFromString('H4-4'))
    expect(s5.board.toString()).toEqual('0,0,0,0,5')
    expect(factory.toString(s5)).toEqual('_|_|_|_|5')
    expect(factory.legalActions(s5).map(a => support.actionToString(a))).toEqual(
      ['H5-1', 'H5-2', 'H5-3', 'H5-4', 'H5-5']
    )
    expect(factory.terminal(s5)).toEqual(false)
    expect(factory.reward(s5, s5.player) === 0).toBeTruthy()
    const s6 = factory.step(s5, support.actionFromString('H5-4'))
    expect(s6.board.toString()).toEqual('0,0,0,0,1')
    expect(factory.toString(s6)).toEqual('_|_|_|_|1')
    expect(factory.legalActions(s6).map(a => support.actionToString(a))).toEqual(
      ['H5-1']
    )
    expect(factory.terminal(s6)).toEqual(true)
    expect(factory.reward(s6, s6.player)).toEqual(-1)
    const s7 = factory.step(s6, support.actionFromString('H5-1'))
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
    expect(history.map((a: number) => support.actionToString({id: a}))).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
    const deserializedState = factory.deserialize(factory.serialize(s4))
    expect(deserializedState.player).toEqual(1)
    expect(deserializedState.board).toEqual([0, 0, 0, 4, 5])
    expect(deserializedState.history.map((a: Action) => support.actionToString(a))).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
  })
  test('Check specific Nim Game', async () => {
    const state = factory.reset()
    expect(state.board.toString()).toEqual('1,2,3,4,5')
    const s1 = factory.step(state, support.actionFromString('H3-1'))
    expect(s1.board.toString()).toEqual('1,2,2,4,5')
    const s2 = factory.step(s1, support.actionFromString('H5-2'))
    expect(s2.board.toString()).toEqual('1,2,2,4,3')
    const s3 = factory.step(s2, support.actionFromString('H4-2'))
    expect(s3.board.toString()).toEqual('1,2,2,2,3')
    const s4 = factory.step(s3, support.actionFromString('H1-1'))
    expect(s4.board.toString()).toEqual('0,2,2,2,3')
    const s5 = factory.step(s4, support.actionFromString('H5-1'))
    expect(s5.board.toString()).toEqual('0,2,2,2,2')
    const s6 = factory.step(s5, support.actionFromString('H2-1'))
    expect(s6.board.toString()).toEqual('0,1,2,2,2')
    expect('H3-1,H4-1,H5-1'.includes(support.actionToString(factory.expertAction(s6)))).toBeTruthy()
    tf.tidy(() => {
      const pi = factory.expertActionPolicy(s6)
      expect(pi.sum().bufferSync().get(0)).toBeGreaterThan(0.9999)
      expect(pi.max().bufferSync().get(0)).toEqual(0.2823789119720459)
      expect(pi.min().bufferSync().get(0)).toEqual(0)
    })
    const s7 = factory.step(s6, support.actionFromString('H3-1'))
    expect(s7.board.toString()).toEqual('0,1,1,2,2')
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
