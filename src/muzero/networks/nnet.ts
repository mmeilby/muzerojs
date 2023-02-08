import * as tf from '@tensorflow/tfjs-node'
import {NetworkOutput} from "./networkoutput";
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise} from "../selfplay/entities";

export interface MuZeroObservation {
}

export interface MuZeroHiddenState {
}

export interface MuZeroNetwork<Action> {
  initialInference (obs: MuZeroObservation): NetworkOutput
  recurrentInference (hiddenState: MuZeroHiddenState, action: Action): NetworkOutput
  trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number[]>
  policyTransform (policy: number): tf.Tensor
  save (path: string): Promise<void>
  load (path: string): Promise<void>
  copyWeights (network: MuZeroNetwork<Action>): void
}
