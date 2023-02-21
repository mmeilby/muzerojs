import * as tf from '@tensorflow/tfjs-node'
import { scalarToSupport, supportToScalar } from './utils'
import { NetworkOutput } from './networkoutput'
import { Batch } from '../replaybuffer/batch'
import { Network, Observation } from './nnet'
import { Actionwise } from '../games/core/actionwise'
import debugFactory from 'debug'
import { Config } from '../games/core/config'

const debug = debugFactory('alphazero:network:debug')

export class MuZeroNetObservation implements Observation {
  constructor (
    public state: number[][]
  ) {}
}

export class NNet<Action extends Actionwise> implements Network<Action> {
  // Length of the hidden state tensors (number of outputs for g.s and h.s)
  protected readonly hxSize: number
  // Length of the reward representation tensors (number of bins)
  protected readonly rewardSupportSize: number
  // Length of the value representation tensors (number of bins)
  protected readonly valueSupportSize: number
  // Size of hidden layer
  public readonly hiddenLayerSize: number

  // Scale the value loss to avoid over fitting of the value function,
  // paper recommends 0.25 (See paper appendix Reanalyze)
  private readonly valueScale: number
  // L2 weights regularization
  protected readonly weightDecay: number

  private readonly inferenceModel: tf.LayersModel

  private readonly logDir: string

  constructor (
    private readonly config: Config
  ) {
    // hidden state size
    this.hxSize = 32
    this.rewardSupportSize = 0
    this.valueSupportSize = 0
    this.hiddenLayerSize = 128
    this.valueScale = 0.25
    this.weightDecay = 0.0001

    this.logDir = './logs/alphazero'

    const channels = 512
    const dropout = 0.3
    const observationInput = tf.input({ shape: config.observationSize, name: 'observation_input' })
    const reshapedInput = tf.layers.reshape({
      targetShape: [this.config.observationSize[0], this.config.observationSize[1], 1]
    }).apply(observationInput) as tf.SymbolicTensor
    const conv1 = this.resildualBlock('conv1', channels, reshapedInput)
    //    const conv2 = this.resildualBlock('conv2', channels, conv1)
    //    const conv3 = this.resildualBlock('conv3', channels, conv2)
    //    const conv4 = this.resildualBlock('conv4', channels, conv3)
    const flatten = tf.layers.flatten().apply(conv1)
    const dense1 = tf.layers.dense({ units: 1024 }).apply(flatten)
    const bn1 = tf.layers.batchNormalization({ axis: 1 }).apply(dense1)
    const relu1 = tf.layers.activation({ activation: 'relu' }).apply(bn1)
    const fc1 = tf.layers.dropout({ rate: dropout }).apply(relu1)
    const dense2 = tf.layers.dense({ units: 512 }).apply(fc1)
    const bn2 = tf.layers.batchNormalization({ axis: 1 }).apply(dense2)
    const relu2 = tf.layers.activation({ activation: 'relu' }).apply(bn2)
    const fc2 = tf.layers.dropout({ rate: dropout }).apply(relu2)

    this.inferenceModel = tf.model({
      name: 'AlphaZero Model',
      inputs: observationInput,
      outputs: [
        tf.layers.dense({ name: 'prediction_policy_output', units: config.actionSpace, activation: 'softmax' }).apply(fc2) as tf.SymbolicTensor,
        tf.layers.dense({ name: 'prediction_value_output', units: 1, activation: 'tanh' }).apply(fc2) as tf.SymbolicTensor
      ]
    })
    this.inferenceModel.compile({
      optimizer: tf.train.adam(config.lrInit),
      loss: {
        prediction_policy_output: tf.losses.softmaxCrossEntropy,
        prediction_value_output: tf.losses.meanSquaredError
      },
      metrics: ['acc']
    })
  }

  private resildualBlock (name: string, channels: number, input: tf.SymbolicTensor): tf.SymbolicTensor {
    const conv = tf.layers.conv2d({
      name: `rb_${name}_CONV2D`,
      kernelSize: 3,
      strides: 1,
      filters: channels,
      padding: 'same'
    }).apply(input) as tf.SymbolicTensor
    const bn = tf.layers.batchNormalization({
      name: `rb_${name}_BN`,
      axis: 3
    }).apply(conv) as tf.SymbolicTensor
    return tf.layers.reLU({
      name: `rb_${name}_RELU`
    }).apply(bn) as tf.SymbolicTensor
  }

  private makeHiddenLayer (name: string, units: number): tf.layers.Layer {
    return tf.layers.dense({
      name,
      units,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
  }

  // Representation network: h(obs)->state
  private h (): { sh: tf.layers.Layer, s: tf.layers.Layer } {
    const hs = tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'linear',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      sh: this.makeHiddenLayer('representation_state_hidden', this.hiddenLayerSize),
      s: hs
    }
  }

  // Prediction network: f(state)->policy,value
  private f (): { vh: tf.layers.Layer, v: tf.layers.Layer, ph: tf.layers.Layer, p: tf.layers.Layer } {
    const fv = tf.layers.dense({
      name: 'prediction_value_output',
      units: 1,
      activation: 'tanh',
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    const fp = tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.config.actionSpace,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      vh: this.makeHiddenLayer('prediction_value_hidden', this.hiddenLayerSize),
      v: fv,
      ph: this.makeHiddenLayer('prediction_policy_hidden', this.hiddenLayerSize),
      p: fp
    }
  }

  /**
   * initialInference
   * Execute h(o)->s f(s)->p,v
   * @param obs
   */
  public initialInference (obs: MuZeroNetObservation): NetworkOutput {
    const observation = tf.tensor2d(obs.state)
    const [tfPolicy, tfValue] = this.inferenceModel.predict(observation.expandDims(0)) as tf.Tensor[]
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    observation.dispose()
    tfPolicy.dispose()
    tfValue.dispose()
    return new NetworkOutput(value, policy)
  }

  /**
   * trainInference
   * @param samples
   */
  public async trainInference (samples: Array<Batch<Actionwise>>): Promise<number> {
    debug(`Training initial batch of size=${samples.length}`)

    const rObservations = tf.tidy(() => tf.concat(samples.reduce((ar, batch) => ar.concat(batch.image.map(img => tf.tensor2d((img as MuZeroNetObservation).state).expandDims(0))), new Array<tf.Tensor<tf.Rank>>())))
    const rTargetPolicies = tf.tidy(() => tf.concat(samples.reduce((ar, batch) => ar.concat(batch.targets.map(target => this.policyPredict(target.policy))), new Array<tf.Tensor<tf.Rank>>())))
    const rTargetRewards = tf.tidy(() => tf.concat(samples.reduce((ar, batch) => ar.concat(batch.targets.map(target => this.valueTransform(target.reward))), new Array<tf.Tensor<tf.Rank>>())))

    debug(`Training recurrent batch of size=${rObservations.shape[0]}`)
    /*
      Use:
        pip install tensorboard  # Unless you've already installed it.
        C:\Users\Morten\AppData\Local\Programs\Python\Python39\Scripts\tensorboard.exe --logdir ./logs/fit_logs_1
     */
    const history = await this.inferenceModel.fit(
      rObservations,
      {
        prediction_policy_output: rTargetPolicies,
        prediction_value_output: rTargetRewards
      },
      {
        batchSize: Math.floor(this.config.batchSize / this.config.gradientUpdateFreq),
        epochs: this.config.epochs,
        verbose: 0,
        shuffle: true,
        validationSplit: this.config.validationSize / this.config.batchSize,
        callbacks: tf.node.tensorBoard(this.logDir, { updateFreq: 'epoch', histogramFreq: 0 })
      }
    )
    rObservations.dispose()
    rTargetPolicies.dispose()
    rTargetRewards.dispose()
    const loss = history.history.loss[0] as number
    return loss
  }

  private inversePolicyTransform (x: tf.Tensor): number[] {
    return x.squeeze().arraySync() as number[]
  }

  private policyTransform (policy: number): tf.Tensor {
    // One hot encode integer actions to Tensor2D
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.config.actionSpace, 1, 0, 'float32')
    //    const tfPolicy = tf.softmax(onehot)
    //    debug(`PolicyTransform: oneHot=${onehot.toString()}, policy=${tfPolicy.toString()}`)
    //    return tfPolicy
  }

  private policyPredict (policy: number[]): tf.Tensor {
    // define the probability for each action based on popularity (visits)
    return tf.softmax(tf.tensor1d(policy)).expandDims(0)
  }

  private inverseRewardTransform (rewardLogits: tf.Tensor): number {
    return supportToScalar(rewardLogits, this.rewardSupportSize)[0]
  }

  private inverseValueTransform (valueLogits: tf.Tensor): number {
    return supportToScalar(valueLogits, this.valueSupportSize)[0]
  }

  /*
  private rewardTransform (reward: number): tf.Tensor {
    return scalarToSupport([reward], this.rewardSupportSize)
  }
*/
  private valueTransform (value: number): tf.Tensor {
    return scalarToSupport([value], this.valueSupportSize)
  }

  public async save (path: string): Promise<void> {
    await Promise.all([
      this.inferenceModel.save(path + 'ii')
    ])
  }

  public async load (path: string): Promise<void> {
    const [
      initialInference
    ] = await Promise.all([
      tf.loadLayersModel(path + 'ii/model.json')
    ])
    this.inferenceModel.setWeights(initialInference.getWeights())
    initialInference.dispose()
  }

  public copyWeights (network: Network<Action>): void {
    tf.tidy(() => {
      network.inferenceModel.setWeights(this.inferenceModel.getWeights())
    })
  }
}
