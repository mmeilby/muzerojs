import { Tensor } from '@tensorflow/tfjs'
import { MuZeroTarget } from './target'

export class MuZeroBatch<Action> {
  constructor (
    public readonly image: Tensor,
    public readonly actions: Action[],
    public readonly targets: MuZeroTarget[]
  ) {}
}
