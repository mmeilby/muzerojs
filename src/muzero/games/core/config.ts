export class Config {

  // ---------------------------------
  // Model configuration

  // Number of all possible actions
  public readonly actionSpace: number
  // Size of the board representation used as input for the representation network - h(obs)
  public readonly observationSize: number[]

  // ---------------------------------
  // Replay buffer configuration

  // Number of self-play games to keep in the replay buffer
  public replayBufferSize: number = 500
  // Number of parts of games to train on at each training step
  public batchSize: number = 64
  // Number of game moves to keep for every batch element
  public numUnrollSteps: number = 10
  // Number of steps in the future to take into account for calculating the target value
  public tdSteps: number

  // --------------------------------
  // Self play configuration

  // Total number of self play steps
  public selfPlaySteps: number = 1000 // TODO: Remove this item
  // Number of future moves self-simulated in MCTS
  public simulations: number = 100
  // Maximum number of moves if game is not finished before
  public maxMoves: number
  // Number of previous observations and previous actions to add to the current observation
  public stackedObservations: number = 10 // TODO: Remove this configuration
  // Prioritized Replay (See paper appendix Training),
  // select in priority the elements in the replay buffer which are unexpected for the network
  public prioritizedReplay: boolean = false
  // How much prioritization is used, 0 corresponding to the uniform case,
  // paper suggests 1.0
  public priorityAlpha: number = 1.0
  // Chronological discount of the reward. Defaults to 1.0
  public discount: number = 1.0
  // The multiplier by which to decay the reward in the backpropagtion phase. Defaults to 1.0
  public decayingParam: number = 1.0
  // Exploration noise to include when exploring possible actions.
  // In order to ensure that the Monte Carlo Tree Search explores a range of possible actions
  // rather than only exploring the action which it currently believes to be optimal.
  // For chess, rootDirichletAlpha = 0.3, defaults to 0.15
  public rootDirichletAlpha: number = 0.15
  // The fraction of noise to include when exploring possible actions.
  // A fraction of 0 disables noise. A fraction of 1 suppress optimal nodes search.
  // Defaults to 0.25 (25% noise included)
  public rootExplorationFraction: number = 0.25
  // UCB hyperparameter range - choose large value to reduce the impact of number of visits on the exploration/exploitation trade-off
  // A base equal to number of simulations will restrain c to ]cinit; cinit + 0,7[
  // Larger values reduce the range towards cinit
  public pbCbase: number = 19652
  // UCB hyperparameter base - the lower boundary for the hyperparameter - should be in the range [1.0; 2.0]
  public pbCinit: number = 1.25

  // -------------------------------------
  // Training configuration

  // Total number of training steps (ie weights update according to a batch)
  public trainingSteps: number = 1000
  // Number of training steps before using the model for self-playing
  public checkpointInterval: number = 25
  // Game specific path for saved network weights
  public savedNetworkPath: string = ''
  // L2 weights regularization
  public weightDecay: number = 0.0001
  // Used only if optimizer is SGD
  public momentum: number = 0.9
  // Initial learning rate
  public lrInit: number = 0.001
  // Exponential learning rate schedule
  // Set it to 1 to use a constant learning rate
  public lrDecayRate: number = 1.0
  // Number of steps to decay the learning rate?
  public lrDecaySteps: number = 10000

  /**
   * Construct the configuration object
   * @param actionSpace Number of all possible actions
   * @param observationSize Size of the board representation used as input
   * for the representation network - h(obs)
   */
  constructor (actionSpace: number, observationSize: number[]) {
    this.actionSpace = actionSpace
    this.tdSteps = actionSpace
    this.maxMoves = actionSpace
    this.observationSize = observationSize
  }
}
