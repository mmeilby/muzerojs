import { describe, test, expect } from '@jest/globals'
import { MuZeroNim } from '../../muzero/games/nim/nim'
import { NimNetModel } from '../../muzero/games/nim/nimmodel'
import { MuZeroConfig } from '../../muzero/games/core/config'
import { type MuZeroNetObservation } from '../../muzero/networks/network'
import { MuZeroAction } from '../../muzero/games/core/action'
import { MuZeroNimUtil } from '../../muzero/games/nim/nimutil'

describe('Nim Unit Test:', () => {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const support = new MuZeroNimUtil()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
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
    const obs = model.observation(state) as MuZeroNetObservation
    expect(obs.observation.toString()).toEqual('1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,1')
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
//    expect(factory.expertActionEnhanced(s2)).toEqual([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
    expect(factory.terminal(s7)).toBeTruthy()
    expect(s7.player).toEqual(-1)
    expect(factory.reward(s7, s7.player)).toEqual(1)
    expect(factory.reward(s7, -s7.player)).toEqual(-1)
    const [player, board, history] = JSON.parse(factory.serialize(s4))
    expect(player).toEqual(1)
    expect(board).toEqual([0, 0, 0, 4, 5])
    expect(history.map((a: number) => support.actionToString(new MuZeroAction(a)))).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
    const deserializedState = factory.deserialize(factory.serialize(s4))
    expect(deserializedState.player).toEqual(1)
    expect(deserializedState.board).toEqual([0, 0, 0, 4, 5])
    expect(deserializedState.history.map((a: MuZeroAction) => support.actionToString(a))).toEqual(['H1-1', 'H3-3', 'H2-1', 'H2-1'])
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
    const policy = factory.expertActionPolicy(s6)
    expect(policy.filter(p => p > 0).length).toEqual(3)
    expect(policy[support.actionFromString('H3-1').id]).toEqual(1 / 3)
    expect(policy[support.actionFromString('H4-1').id]).toEqual(1 / 3)
    expect(policy[support.actionFromString('H5-1').id]).toEqual(1 / 3)
    const s7 = factory.step(s6, support.actionFromString('H3-1'))
    expect(s7.board.toString()).toEqual('0,1,1,2,2')
  })
})
