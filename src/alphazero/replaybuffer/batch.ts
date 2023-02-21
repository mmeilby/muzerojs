import { Target } from './target'
import { Observation } from '../networks/nnet'

export class Batch<Action> {
  constructor (
    // Observation image for states in the batch
    public readonly image: Observation[],
    // Targets for each state
    public readonly targets: Target[]
  ) {}
}
