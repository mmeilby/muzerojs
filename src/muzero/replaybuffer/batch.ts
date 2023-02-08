import { MuZeroTarget } from './target'
import {MuZeroObservation} from "../networks/nnet";

export class MuZeroBatch<Action> {
  constructor (
    // Observation image for first state in the batch
    public readonly image: MuZeroObservation,
    // Sequence of actions played for this batch
    public readonly actions: Action[],
    // Targets for each turn played by executing the corresponding action
    public readonly targets: MuZeroTarget[]
  ) {}
}
