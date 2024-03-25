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
  public replayBufferSize: number
  // Number of parts of games to train on at each training step
  public batchSize: number
  // Number of steps in the future to take into account for calculating the target value
  public tdSteps: number
  // Number of game moves to keep for every batch element
  public numUnrollSteps: number
  // Number of previous observations and previous actions
  // to add to the current observation
  public stackedObservations: number
  // Prioritized Replay (See paper appendix Training),
  // select in priority the elements in the replay buffer which are unexpected for the network
  public prioritizedReplay: boolean
  // How much prioritization is used, 0 corresponding to the uniform case,
  // paper suggests 1.0
  public priorityAlpha: number

  // --------------------------------
  // Self play configuration

  // Threshold for temperature switch in episode steps (moves)
  public temperatureThreshold: number
  // Total number of self play games played
  public numEpisodes: number
  // Number of future moves self-simulated in MCTS
  public simulations: number
  // Maximum number of moves if game is not finished before
  public maxMoves: number
  // Chronological discount of the reward. Defaults to 1.0
  public discount: number
  // The multiplier by which to decay the reward in the backpropagtion phase. Defaults to 1.
  public decayingParam: number
  // Exploration noise to include when exploring possible actions.
  // In order to ensure that the Monte Carlo Tree Search explores a range of possible actions
  // rather than only exploring the action which it currently believes to be optimal.
  // For chess, rootDirichletAlpha = 0.3, defaults to 0.15
  public rootDirichletAlpha: number
  // The fraction of noise to include when exploring possible actions.
  // A fraction of 0 disables noise. A fraction of 1 suppress optimal nodes search.
  // Defaults to 0.25 (25% noise included)
  public rootExplorationFraction: number
  // UCB calculation contants
  public pbCbase: number
  // UCB calculation contants
  public pbCinit: number

  // -------------------------------------
  // Training configuration

  // Total number of training steps (model fittings) before arena pitting
  public trainingSteps: number
  // Total number of training iterations for each training step
  public epochs: number
  // Total number data.old sets to be used for validation
  public validationSize: number
  // Number of gradient updates per training step
  public gradientUpdateFreq: number
  // Number of training steps before using the model for self-playing
  public checkpointInterval: number

  // L2 weights regularization
  public weightDecay: number
  // Used only if optimizer is SGD
  public momentum: number

  // Exponential learning rate schedule

  // Initial learning rate
  public lrInit: number
  // Set it to 1 to use a constant learning rate
  public lrDecayRate: number
  // Number of steps to decay the learning rate?
  public lrDecaySteps: number

  // -------------------------------------
  // Arena configuration

  // Threshold for accepting or rejecting a new network
  public networkUpdateThreshold: number
  // Number of games played in arena
  public numGames: number

  // -------------------------------------
  // Coach configuration

  // Number of iterations to generate games, train, and test
  public numIterations: number

  /**
     * Construct the configuration object
     * @param actionSpace Number of all possible actions
     * @param observationSize Size of the board representation used as input
     * for the representation network - h(obs)
     */
  constructor (actionSpace: number, observationSize: number[]) {
    this.actionSpace = actionSpace
    this.observationSize = observationSize
    this.replayBufferSize = 500
    this.batchSize = 64
    this.tdSteps = actionSpace
    this.numUnrollSteps = 10
    this.stackedObservations = 10 // TODO: Remove this configuration
    this.prioritizedReplay = false // TODO: Remove this configuration
    this.priorityAlpha = 1.0 // TODO: Remove this configuration
    this.temperatureThreshold = 15
    this.networkUpdateThreshold = 0.6
    this.numGames = 10
    this.numIterations = 10
    this.numEpisodes = 10
    this.simulations = 100
    this.maxMoves = actionSpace
    this.discount = 1.0
    this.decayingParam = 1.0
    this.rootDirichletAlpha = 0.15
    this.rootExplorationFraction = 0.25
    this.pbCbase = 19652
    this.pbCinit = 1.25
    this.trainingSteps = 5
    this.epochs = 10
    this.validationSize = 16
    this.gradientUpdateFreq = 1
    this.checkpointInterval = 25
    this.weightDecay = 0.0001
    this.momentum = 0.9
    this.lrInit = 0.001
    this.lrDecayRate = 1.0
    this.lrDecaySteps = 10000
  }
}
