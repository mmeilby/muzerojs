import { ObservationModel } from '../games/core/model'
import { Observation } from './nnet'
import { MockedObservation } from './mnetwork'
import {Statewise} from "../games/core/statewise";

export class MockedModel<State extends Statewise> implements ObservationModel<State> {

  get observationSize (): number[] {
    // Observation size is not used for mocked model
    return []
  }

  public observation (state: State): Observation {
    return new MockedObservation(state)
  }
}
