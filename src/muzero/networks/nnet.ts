import { type TensorNetworkOutput } from './networkoutput'
import { type Batch } from '../replaybuffer/batch'
import { type Model } from './model'
import { type NetworkState } from './networkstate'
import { type Action } from '../games/core/action'
import { type LossLog } from './implementations/core'

export interface Network {
  getModel: () => Model
  initialInference: (state: NetworkState) => TensorNetworkOutput
  recurrentInference: (state: NetworkState, action: Action[]) => TensorNetworkOutput
  trainInference: (samples: Batch[]) => LossLog
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Network) => void
  dispose: () => void
}
