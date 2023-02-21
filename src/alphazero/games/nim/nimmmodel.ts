import { ObservationModel } from '../core/model'
import { NimState } from './nimstate'
import { config } from './nimconfig'
import { Observation } from '../../networks/nnet'
import { MockedObservation } from '../../networks/mnetwork'

export class NimNetMockedModel implements ObservationModel<NimState> {
  get observationSize (): number[] {
    return [config.heaps, config.heapSize]
  }

  public observation (state: NimState): Observation {
    return new MockedObservation(state)
  }
}
