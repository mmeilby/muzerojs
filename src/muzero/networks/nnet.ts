import { type TensorNetworkOutput } from './networkoutput'
import { type Batch } from '../replaybuffer/batch'
import { type Model } from './model'
import { NetworkState } from './networkstate'
import { Action } from '../games/core/action'

export interface Network {
  getModel: () => Model
  initialInference: (state: NetworkState) => TensorNetworkOutput
  recurrentInference: (state: NetworkState, action: Action[]) => TensorNetworkOutput
  trainInference: (samples: Batch[]) => number[]
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Network) => void
  dispose: () => void
}
