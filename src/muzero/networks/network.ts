import * as tf from '@tensorflow/tfjs-node'
import { scalarToSupport, supportToScalar } from '../selfplay/utils'
import { NetworkOutput } from './networkoutput'
import { TrainedNetworkOutput } from './trainednetworkoutput'

export abstract class BaseMuZeroNet {
  private readonly zeroReward: tf.Tensor

  // Length of the state tensors
  protected readonly hxSize: number
  // Length of the action tensors
  protected readonly actionSpaceN: number
  // Length of the reward representation tensors (number of bins)
  protected readonly rewardSupportSize: number
  // Length of the value representation tensors (number of bins)
  protected readonly valueSupportSize: number
  // Value loss scale
  private readonly valueScale: number

  private forwardModel: tf.LayersModel
  private recurrentModel: tf.LayersModel

  private readonly logDir: string

  protected abstract h (observationInput: tf.SymbolicTensor): tf.SymbolicTensor
  protected abstract f (stateInput: tf.SymbolicTensor): { v: tf.SymbolicTensor, p: tf.SymbolicTensor }
  protected abstract g (actionPlaneInput: tf.SymbolicTensor): { s: tf.SymbolicTensor, r: tf.SymbolicTensor }

  constructor (
    inputSize: number,
    actionSpace: number
  ) {
    // hidden state size
    this.hxSize = 32
    this.rewardSupportSize = 10
    this.valueSupportSize = 10
    this.valueScale = 0.25

    this.actionSpaceN = actionSpace
    this.zeroReward = tf.oneHot([this.rewardSupportSize], this.rewardSupportSize * 2 + 1)

    this.logDir = './logs/20230109-005200' // + sysdatetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    // make model for h(o)->s f(s)->p,v
    // s: batch_size x time x state_x x state_y
    const observationInput = tf.input({ shape: [inputSize], name: 'observation_input' })
    const h = this.h(observationInput)
    const f1 = this.f(h)
    const iiModel = tf.model({
      name: 'Initial Inference',
      inputs: observationInput,
      outputs: [h, f1.p, f1.v]
    })
    iiModel.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })
    this.forwardModel = iiModel

    // make model for g(s,a)->s,r f(s)->p,v
    // a: one hot encoded vector of shape batch_size x (state_x * state_y)
    const actionPlaneInput = tf.input({ shape: [this.hxSize + this.actionSpaceN], name: 'action_plane_input' })
    const g = this.g(actionPlaneInput)
    const f2 = this.f(g.s)
    const riModel = tf.model({
      name: 'Recurrent Inference',
      inputs: actionPlaneInput,
      outputs: [g.s, g.r, f2.p, f2.v]
    })
    riModel.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })
    this.recurrentModel = riModel
  }

  /**
     * initialInference
     * @param obs
     */
  public initialInference (obs: tf.Tensor): NetworkOutput {
    const result = this.forwardModel.predict(obs.reshape([1, -1])) as tf.Tensor[]
    const tfValue = result[2]
    const value = this.inverseValueTransform(result[2])
    const tfReward = this.rewardTransform(0)
    const reward = 0
    const tfPolicy = result[1]
    const policy = this.inversePolicyTransform(result[1])
    const state = result[0]
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, state)
  }

  /**
     * recurrentInference
     * @param hiddenState
     * @param action
     */
  public recurrentInference (hiddenState: tf.Tensor, action: tf.Tensor): NetworkOutput {
    const x = tf.concat([hiddenState, action], 1)
    const result = this.recurrentModel.predict(x) as tf.Tensor[]
    const tfValue = result[3]
    const value = this.inverseValueTransform(result[3])
    const tfReward = result[1]
    const reward = this.inverseRewardTransform(result[1])
    const tfPolicy = result[2]
    const policy = this.inversePolicyTransform(result[2])
    const state = result[0]
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, state)
  }

  /**
     * trainInitialInference
     * @param obs
     * @param targetPolicy
     * @param targetValue
     */
  public trainInitialInference (obs: tf.Tensor, targetPolicy: number[], targetValue: number): TrainedNetworkOutput {
    let state: tf.Tensor = tf.tensor1d(new Array(this.hxSize))
    const policyGradients = tf.variableGrads(() => {
      const result = this.forwardModel.predict(obs.reshape([1, -1])) as tf.Tensor[]
      state = tf.keep(result[0])
      return tf.losses.softmaxCrossEntropy(this.policyPredict(targetPolicy), result[1]).asScalar().add(
        tf.losses.sigmoidCrossEntropy(this.valueTransform(targetValue), result[2]).asScalar().mul(this.valueScale))
    })
    return {
      grads: policyGradients.grads,
      loss: policyGradients.value,
      state
    }
  }

  /**
     * trainRecurrentInference
     * @param hiddenState
     * @param action
     * @param targetPolicy
     * @param targetValue
     * @param targetReward
     * @param lossScale
     */
  public trainRecurrentInference (hiddenState: tf.Tensor, action: tf.Tensor, targetPolicy: number[], targetValue: number, targetReward: number, lossScale: number): TrainedNetworkOutput {
    const x = tf.concat([hiddenState, action], 1)
    let state: tf.Tensor = tf.tensor1d(new Array(this.hxSize))
    const policyGradients = tf.variableGrads(() => {
      const result = this.recurrentModel.predict(x) as tf.Tensor[]
      state = tf.keep(result[0])
      return tf.losses.sigmoidCrossEntropy(this.rewardTransform(targetReward), result[1]).asScalar().add(
        tf.losses.softmaxCrossEntropy(this.policyPredict(targetPolicy), result[2]).asScalar().add(
          tf.losses.sigmoidCrossEntropy(this.valueTransform(targetValue), result[3]).asScalar().mul(this.valueScale))).mul(lossScale)
    })
    return {
      grads: policyGradients.grads,
      loss: policyGradients.value,
      state
    }
  }

  public inversePolicyTransform (x: tf.Tensor): number[] {
    return tf.softmax(x).squeeze().arraySync() as number[]
  }

  public policyTransform (policy: number): tf.Tensor {
    // One hot encode integer actions to Tensor2D
    return tf.cast(tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN), 'float32')
  }

  public policyPredict (policy: number[]): tf.Tensor {
    // define the probability for each action based on popularity (visits)
    const probs = tf.softmax(tf.tensor1d(policy))
    // select the most popular action
    const action = tf.multinomial(probs, 1, undefined, false) as tf.Tensor1D
    // One hot encode action to Tensor2D
    return tf.cast(tf.oneHot(tf.tensor1d([action.bufferSync().get(0)], 'int32'), this.actionSpaceN), 'float32')
  }

  public inverseRewardTransform (rewardLogits: tf.Tensor): number {
    return supportToScalar(rewardLogits, this.rewardSupportSize)[0]
  }

  public inverseValueTransform (valueLogits: tf.Tensor): number {
    return supportToScalar(valueLogits, this.valueSupportSize)[0]
  }

  public rewardTransform (reward: number): tf.Tensor {
    return scalarToSupport([reward], this.rewardSupportSize)
  }

  public valueTransform (value: number): tf.Tensor {
    return scalarToSupport([value], this.valueSupportSize)
  }

  public async save (path: string): Promise<void> {
    await Promise.all([
      this.forwardModel.save(path + '_forward'),
      this.recurrentModel.save(path + '_recurrent')
    ])
  }

  public async load (path: string): Promise<void> {
    const models = await Promise.all([
      tf.loadLayersModel(path + '_forward/model.json'),
      tf.loadLayersModel(path + '_recurrent/model.json')
    ])
    this.forwardModel = models[0]
    this.recurrentModel = models[1]
  }

  public trainingSteps (): number {
    return 1
  }
}
