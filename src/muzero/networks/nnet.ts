import { NetworkOutput } from './networkoutput'
import { MuZeroBatch } from '../replaybuffer/batch'
import { Actionwise } from '../selfplay/entities'

export interface MuZeroObservation {
}

export interface MuZeroHiddenState {
}

export interface MuZeroNetwork<Action> {
  initialInference: (obs: MuZeroObservation) => NetworkOutput
  recurrentInference: (hiddenState: MuZeroHiddenState, action: Action) => NetworkOutput
  trainInference: (samples: Array<MuZeroBatch<Actionwise>>) => Promise<number[]>
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: MuZeroNetwork<Action>) => void
}
