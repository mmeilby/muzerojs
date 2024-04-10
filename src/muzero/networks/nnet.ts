import { type NetworkOutput } from './networkoutput'
import { type Batch } from '../replaybuffer/batch'
import {Action} from "../selfplay/mctsnode";

export interface Observation {
}

export interface HiddenState {
}

export interface Network {
  initialInference (obs: Observation): NetworkOutput
  recurrentInference (hiddenState: HiddenState, action: Action): NetworkOutput
  trainInference (samples: Array<Batch>): number[]
  save (path: string): Promise<void>
  load (path: string): Promise<void>
  copyWeights (network: Network): void
}
