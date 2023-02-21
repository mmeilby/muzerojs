import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { Network, Observation } from './nnet'
import { Environment } from '../games/core/environment'
import { Actionwise } from '../games/core/actionwise'
import { Statewise } from '../games/core/statewise'

export class MockedObservation<State> implements Observation {
  constructor (
    public state: State
  ) {}
}

export class MockedNetwork<State extends Statewise, Action extends Actionwise> implements Network<Action> {
  // Length of the action tensors
  protected readonly actionSpaceN: number

  constructor (
    private readonly env: Environment<State, Action>
  ) {
    this.actionSpaceN = env.config().actionSpaceSize
  }

  public initialInference (obs: MockedObservation<State>): NetworkOutput {
    // The mocked network will respond with the perfect move
    const policy = this.env.expertAction(obs.state)
    const value = this.env.reward(obs.state, obs.state.player)
    return new NetworkOutput(value, policy)
  }

  public async trainInference (samples: Array<Batch<Actionwise>>): Promise<number> {
    // Return the perfect loss and accuracy vector
    return await Promise.resolve(0)
  }

  public async save (path: string): Promise<void> {
    // No reason for saving anything from a mocked network
  }

  public async load (path: string): Promise<void> {
    // We can't load any data to a mocked network - ignore
  }

  public copyWeights (network: Network<Action>): void {
    // A mocked network does not have any data to copy - leave the target network untouched
  }
}
