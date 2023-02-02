import { describe, test, expect } from '@jest/globals'
import { MuZeroSharedStorage } from '../muzero/training/sharedstorage'
import { MuZeroAction } from '../muzero/games/core/action'
import { MuZeroReplayBuffer } from '../muzero/replaybuffer/replaybuffer'
import { MuZeroSelfPlay } from '../muzero/selfplay/selfplay'
import { MuZeroNim } from '../muzero/games/nim/nim'
import { MuZeroNimState } from '../muzero/games/nim/nimstate'
import { NimNetModel } from '../muzero/games/nim/nimmodel'
import { MuZeroNet } from '../muzero/networks/fullconnected'
import {MuZeroConfig} from "../muzero/games/core/config";

describe('Muzero Self Play Unit Test:', () => {
  const factory = new MuZeroNim()
  const model = new NimNetModel()
  const config = factory.config()
  const conf = new MuZeroConfig(config.actionSpaceSize, model.observationSize)
  test('Check the Nim Game', async () => {
    const state = factory.reset()
    expect(state.board.toString()).toEqual('1,2,3')
    expect(factory.toString(state)).toEqual('1|2|3')
    expect(factory.legalActions(state).map(a => a.id).toString()).toEqual('0,3,4,6,7,8')
    const a = model.observation(state)
    expect(a.toString()).toEqual('1,0,0,1,1,0,1,1,1')
    const s1 = factory.step(state, new MuZeroAction(0))
    expect(s1.board.toString()).toEqual('0,2,3')
    expect(factory.legalActions(s1).map(a => a.id).toString()).toEqual('3,4,6,7,8')
    const s2 = factory.step(s1, new MuZeroAction(8))
    expect(s2.board.toString()).toEqual('0,2,0')
    expect(factory.legalActions(s2).map(a => a.id).toString()).toEqual('3,4')
    expect(factory.expertAction(s2).id).toEqual(3)
    const s3 = factory.step(s2, new MuZeroAction(3))
    expect(s3.board.toString()).toEqual('0,1,0')
    expect(factory.toString(s3)).toEqual('_|1|_')
    expect(factory.legalActions(s3).map(a => a.id).toString()).toEqual('3')
    expect(factory.expertAction(s3).id).toEqual(3)
    expect(factory.terminal(s3)).toEqual(true)
    expect(factory.reward(s3, s3.player)).toEqual(1)
    const s4 = factory.step(s3, new MuZeroAction(3))
    expect(s4.board.toString()).toEqual('0,0,0')
    expect(factory.legalActions(s4).length).toEqual(0)
    expect(factory.expertAction(s4).id).toEqual(-1)
    expect(factory.terminal(s4)).toEqual(true)
    expect(factory.reward(s4, s4.player)).toEqual(1)
    expect(factory.reward(s4, -s4.player)).toEqual(-1)
    expect(factory.serialize(s4)).toEqual('[1,[0,0,0],[0,8,3,3]]')
    expect(factory.serialize(factory.deserialize(factory.serialize(s4)))).toEqual('[1,[0,0,0],[0,8,3,3]]')
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
