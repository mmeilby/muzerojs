/**
 *
 * @hidden
 * @internal
 * @param State An type representing the state of the game.
 */
export class MCTSState<State> {
  // The predicted reward received by moving to this state
  private es_: number
  // The number of times state s has been visited
  private ns_: number
  // N(s,a) - the number of times we took action a from state s across simulation
  private nsa_: number
  // P(s,a) - the predicted probability of choosing the action that leads to this state
  private psa_: number
  // The predicted backfilled value average of the node
  private qsa_: number

  constructor (
    private readonly state_: State
  ) {
    this.es_ = 0
    this.ns_ = 0
    this.nsa_ = 0
    this.psa_ = 0
    this.qsa_ = 0
  }

  /**
     * Q(s,a) - The expected reward for taking action a from state s
     */
  get Qsa (): number {
    return this.qsa_
  }

  set Qsa (value: number) {
    this.qsa_ = value
  }

  /**
     * The number of times this node has been visited (updated on each back propagation) - N(s,a)
     */
  get Nsa (): number {
    return this.nsa_
  }

  set Nsa (value: number) {
    this.nsa_ = value
  }

  get Ns (): number {
    return this.ns_
  }

  set Ns (value: number) {
    this.ns_ = value
  }

  /**
     * The predicted prior probability of choosing the action that leads to this node - P(s,a)
     */
  get Psa (): number {
    return this.psa_
  }

  set Psa (value: number) {
    this.psa_ = value
  }

  get Es (): number {
    return this.es_
  }

  set Es (value: number) {
    this.es_ = value
  }

  get state (): State {
    return this.state_
  }
}
