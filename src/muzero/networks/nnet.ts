import type * as tf from '@tensorflow/tfjs-node'
import { type TensorNetworkOutput } from './networkoutput'
import { type Batch } from '../replaybuffer/batch'
import { type Model } from './model'

export interface Network {
  getModel: () => Model
  initialInference: (observation: tf.Tensor) => TensorNetworkOutput
  recurrentInference: (hiddenState: tf.Tensor, action: tf.Tensor) => TensorNetworkOutput
  trainInference: (samples: Batch[]) => number[]
  save: (path: string) => Promise<void>
  load: (path: string) => Promise<void>
  copyWeights: (network: Network) => void
  dispose: () => void
}
