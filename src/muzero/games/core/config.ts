export class MuZeroConfig {
  // ---------------------------------
  // Model configuration

  // Number of all possible actions
  private readonly _actionSpace: number
  // Size of the board representation used as input for the representation network - h(obs)
  private readonly _observationSize: number

  // ---------------------------------
  // Replay buffer configuration

  // Number of self-play games to keep in the replay buffer
  private _replayBufferSize?: number
  // Number of parts of games to train on at each training step
  private _batchSize?: number
  // Number of steps in the future to take into account for calculating the target value
  private _tdSteps?: number
  // Number of game moves to keep for every batch element
  private _numUnrollSteps?: number
  // Number of previous observations and previous actions
  // to add to the current observation
  private _stackedObservations?: number
  // Prioritized Replay (See paper appendix Training),
  // select in priority the elements in the replay buffer which are unexpected for the network
  private _prioritizedReplay?: boolean
  // How much prioritization is used, 0 corresponding to the uniform case,
  // paper suggests 1.0
  private _priorityAlpha?: number

  // --------------------------------
  // Self play configuration

  // Total number of self play steps
  private _selfPlaySteps?: number
  // Number of future moves self-simulated in MCTS
  private _simulations?: number
  // Maximum number of moves if game is not finished before
  private _maxMoves?: number
  // Chronological discount of the reward. Defaults to 1.0
  private _discount?: number
  // The multiplier by which to decay the reward in the backpropagtion phase. Defaults to 1.
  private _decayingParam?: number
  // Exploration noise to include when exploring possible actions.
  // In order to ensure that the Monte Carlo Tree Search explores a range of possible actions
  // rather than only exploring the action which it currently believes to be optimal.
  // For chess, rootDirichletAlpha = 0.3, defaults to 0.15
  private _rootDirichletAlpha?: number
  // The fraction of noise to include when exploring possible actions.
  // A fraction of 0 disables noise. A fraction of 1 suppress optimal nodes search.
  // Defaults to 0.25 (25% noise included)
  private _rootExplorationFraction?: number
  // UCB calculation contants
  private _pbCbase?: number
  // UCB calculation contants
  private _pbCinit?: number

  // -------------------------------------
  // Training configuration

  // Total number of training steps (ie weights update according to a batch)
  private _trainingSteps?: number
  // Number of training steps before using the model for self-playing
  private _checkpointInterval?: number

  // L2 weights regularization
  private _weightDecay?: number
  // Used only if optimizer is SGD
  private _momentum?: number

  // Exponential learning rate schedule

  // Initial learning rate
  private _lrInit?: number
  // Set it to 1 to use a constant learning rate
  private _lrDecayRate?: number
  // Number of steps to decay the learning rate?
  private _lrDecaySteps?: number

  /**
     * Construct the configuration object
     * @param actionSpace Number of all possible actions
     * @param observationSize Size of the board representation used as input
     * for the representation network - h(obs)
     */
  constructor (actionSpace: number, observationSize: number) {
    this._actionSpace = actionSpace
    this._observationSize = observationSize
  }

  get replayBufferSize (): number {
    return this._replayBufferSize ?? 500
  }

  get batchSize (): number {
    return this._batchSize ?? 64
  }

  get tdSteps (): number {
    return this._tdSteps ?? this._actionSpace
  }

  get numUnrollSteps (): number {
    return this._numUnrollSteps ?? 10
  }

  get stackedObservations (): number {
    return this._stackedObservations ?? 10 // TODO: Remove this configuration
  }

  get prioritizedReplay (): boolean {
    return this._prioritizedReplay ?? false
  }

  get priorityAlpha (): number {
    return this._priorityAlpha ?? 1.0
  }

  get selfPlaySteps (): number {
    return this._selfPlaySteps ?? 1000
  }

  get simulations (): number {
    return this._simulations ?? 100
  }

  get maxMoves (): number {
    return this._maxMoves ?? this._actionSpace
  }

  get discount (): number {
    return this._discount ?? 1.0
  }

  get decayingParam (): number {
    return this._decayingParam ?? 1.0
  }

  get rootDirichletAlpha (): number {
    return this._rootDirichletAlpha ?? 0.15
  }

  get rootExplorationFraction (): number {
    return this._rootExplorationFraction ?? 0.25
  }

  get pbCbase (): number {
    return this._pbCbase ?? 19652
  }

  get pbCinit (): number {
    return this._pbCinit ?? 1.25
  }

  get trainingSteps (): number {
    return this._trainingSteps ?? 1000
  }

  get checkpointInterval (): number {
    return this._checkpointInterval ?? 25
  }

  get weightDecay (): number {
    return this._weightDecay ?? 0.0001
  }

  get momentum (): number {
    return this._momentum ?? 0.9
  }

  get lrInit (): number {
    return this._lrInit ?? 0.001
  }

  get lrDecayRate (): number {
    return this._lrDecayRate ?? 1.0
  }

  get lrDecaySteps (): number {
    return this._lrDecaySteps ?? 10000
  }

  get actionSpace (): number {
    return this._actionSpace
  }

  get observationSize (): number {
    return this._observationSize
  }

  // ----------- Setters ------------

  set replayBufferSize (value: number) {
    this._replayBufferSize = value
  }

  set batchSize (value: number) {
    this._batchSize = value
  }

  set tdSteps (value: number) {
    this._tdSteps = value
  }

  set numUnrollSteps (value: number) {
    this._numUnrollSteps = value
  }

  set stackedObservations (value: number) {
    this._stackedObservations = value
  }

  set prioritizedReplay (value: boolean) {
    this._prioritizedReplay = value
  }

  set priorityAlpha (value: number) {
    this._priorityAlpha = value
  }

  set selfPlaySteps (value: number) {
    this._selfPlaySteps = value
  }

  set simulations (value: number) {
    this._simulations = value
  }

  set maxMoves (value: number) {
    this._maxMoves = value
  }

  set discount (value: number) {
    this._discount = value
  }

  set decayingParam (value: number) {
    this._decayingParam = value
  }

  set rootDirichletAlpha (value: number) {
    this._rootDirichletAlpha = value
  }

  set rootExplorationFraction (value: number) {
    this._rootExplorationFraction = value
  }

  set pbCbase (value: number) {
    this._pbCbase = value
  }

  set pbCinit (value: number) {
    this._pbCinit = value
  }

  set trainingSteps (value: number) {
    this._trainingSteps = value
  }

  set checkpointInterval (value: number) {
    this._checkpointInterval = value
  }

  set weightDecay (value: number) {
    this._weightDecay = value
  }

  set momentum (value: number) {
    this._momentum = value
  }

  set lrInit (value: number) {
    this._lrInit = value
  }

  set lrDecayRate (value: number) {
    this._lrDecayRate = value
  }

  set lrDecaySteps (value: number) {
    this._lrDecaySteps = value
  }
}
