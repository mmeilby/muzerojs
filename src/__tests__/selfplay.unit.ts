import { describe, test, expect } from '@jest/globals'
import { MuZeroSharedStorage } from '../muzero/training/sharedstorage'
import { MuZeroAction } from '../muzero/games/core/action'
import { MuZeroReplayBuffer } from '../muzero/replaybuffer/replaybuffer'
import { MuZeroSelfPlay } from '../muzero/selfplay/selfplay'
import { MuZeroNim } from '../muzero/games/nim/nim'
import { MuZeroNimState } from '../muzero/games/nim/nimstate'
import { NimNetModel } from '../muzero/games/nim/nimmodel'
import { MuZeroConfig } from "../muzero/games/core/config";
import {MuZeroNet, MuZeroNetObservation} from "../muzero/networks/network";

describe('Muzero Self Play Unit Test:', () => {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  test('Check the Nim Game', async () => {
    const state = factory.reset()
    expect(state.board.toString()).toEqual('1,2,3,4,5')
    expect(factory.toString(state)).toEqual('1|2|3|4|5')
    expect(factory.legalActions(state).map(a => a.id).toString()).toEqual('0,5,6,10,11,12,15,16,17,18,20,21,22,23,24')
    const obs = model.observation(state) as MuZeroNetObservation
    expect(obs.state.toString()).toEqual('1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,1')
    const s1 = factory.step(state, new MuZeroAction(0))
    expect(s1.board.toString()).toEqual('0,2,3,4,5')
    expect(factory.legalActions(s1).map(a => a.id).toString()).toEqual('5,6,10,11,12,15,16,17,18,20,21,22,23,24')
    const s2 = factory.step(s1, new MuZeroAction(12))
    expect(s2.board.toString()).toEqual('0,2,0,4,5')
    expect(factory.legalActions(s2).map(a => a.id).toString()).toEqual('5,6,15,16,17,18,20,21,22,23,24')
    expect(factory.expertAction(s2).id).toEqual(5)
    const s3 = factory.step(s2, new MuZeroAction(5))
    expect(s3.board.toString()).toEqual('0,1,0,4,5')
    expect(factory.toString(s3)).toEqual('_|1|_|4|5')
    expect(factory.legalActions(s3).map(a => a.id).toString()).toEqual('5,15,16,17,18,20,21,22,23,24')
    expect(factory.terminal(s3)).toEqual(false)
    const s4 = factory.step(s3, new MuZeroAction(5))
    expect(s4.board.toString()).toEqual('0,0,0,4,5')
    expect(factory.toString(s4)).toEqual('_|_|_|4|5')
    expect(factory.legalActions(s4).map(a => a.id).toString()).toEqual('15,16,17,18,20,21,22,23,24')
    expect(factory.terminal(s4)).toEqual(false)
    expect(factory.reward(s4, s4.player)).toEqual(0)
    const s5 = factory.step(s4, new MuZeroAction(18))
    expect(s5.board.toString()).toEqual('0,0,0,0,5')
    expect(factory.toString(s5)).toEqual('_|_|_|_|5')
    expect(factory.legalActions(s5).map(a => a.id).toString()).toEqual('20,21,22,23,24')
    expect(factory.terminal(s5)).toEqual(false)
    expect(factory.reward(s5, s5.player)).toEqual(0)
    const s6 = factory.step(s5, new MuZeroAction(23))
    expect(s6.board.toString()).toEqual('0,0,0,0,1')
    expect(factory.toString(s6)).toEqual('_|_|_|_|1')
    expect(factory.legalActions(s6).map(a => a.id).toString()).toEqual('20')
    expect(factory.terminal(s6)).toEqual(true)
    expect(factory.reward(s6, s6.player)).toEqual(-1)
    const s7 = factory.step(s6, new MuZeroAction(20))
    expect(s7.board.toString()).toEqual('0,0,0,0,0')
    expect(factory.legalActions(s7).length).toEqual(0)
    expect(factory.expertAction(s7).id).toEqual(-1)
    expect(factory.terminal(s7)).toEqual(true)
    expect(s7.player).toEqual(-1)
    expect(factory.reward(s7, s7.player)).toEqual(1)
    expect(factory.reward(s7, -s7.player)).toEqual(-1)
    expect(factory.serialize(s4)).toEqual('[1,[0,0,0,4,5],[0,12,5,5]]')
    expect(factory.serialize(factory.deserialize(factory.serialize(s4)))).toEqual('[1,[0,0,0,4,5],[0,12,5,5]]')
  })
  test('Check self play', async () => {
    const replayBuffer = new MuZeroReplayBuffer<MuZeroNimState, MuZeroAction>(conf)
    const sharedStorage = new MuZeroSharedStorage(conf)
    const network = new MuZeroNet(model.observationSize, config.actionSpaceSize, 0.01)
    await sharedStorage.saveNetwork(1, network)
    conf.selfPlaySteps = 2
    const selfPlay = new MuZeroSelfPlay(conf, factory, model)
    await selfPlay.runSelfPlay(sharedStorage, replayBuffer)
    expect(replayBuffer.numPlayedGames).toEqual(2)
  })
})
