import { type NetworkOutput } from './networkoutput'
import { type MuZeroBatch } from '../replaybuffer/batch'
import { type Actionwise } from '../selfplay/entities'

export interface MuZeroObservation {
}

export interface MuZeroHiddenState {
}

export interface MuZeroNetwork<Action> {
  initialInference: (obs: MuZeroObservation) => NetworkOutput
  recurrentInference: (hiddenState: MuZeroHiddenState, action: Action) => NetworkOutput
  trainInference: (samples: Array<MuZeroBatch<Actionwise>>) => number[]
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: MuZeroNetwork<Action>) => void
}
