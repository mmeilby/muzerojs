import { Target } from './target'
import { Observation } from '../networks/nnet'
import {Action} from "../selfplay/mctsnode";

export class Batch {
  constructor (
    // Observation image for first state in the batch
    public readonly image: Observation,
    // Sequence of actions played for this batch
    public readonly actions: Action[],
    // Targets for each turn played by executing the corresponding action
    public readonly targets: Target[]
  ) {}
}
