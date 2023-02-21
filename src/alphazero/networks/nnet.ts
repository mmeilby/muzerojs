import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { Actionwise } from '../games/core/actionwise'

export interface Observation {
}

export interface HiddenState {
}

export interface Network<Action extends Actionwise> {
  initialInference: (obs: Observation) => NetworkOutput
  trainInference: (samples: Array<Batch<Actionwise>>) => Promise<number>
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Network<Action>) => void
}
