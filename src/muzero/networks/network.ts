import * as tf from '@tensorflow/tfjs-node'
import { scalarToSupport, supportToScalar } from '../selfplay/utils'
import { NetworkOutput } from './networkoutput'
import { TrainedNetworkOutput } from './trainednetworkoutput'
import debugFactory from 'debug'
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise} from "../selfplay/entities";
import {MuZeroTarget} from "../replaybuffer/target";
import {MuZeroAction} from "../games/core/action";
import {data} from "@tensorflow/tfjs-node";

const debug = debugFactory('muzero:network:debug')

export abstract class BaseMuZeroNet {
  // Length of the hidden state tensors
  protected readonly hxSize: number
  // Length of the action tensors
  protected readonly actionSpaceN: number
  // Length of the reward representation tensors (number of bins)
  protected readonly rewardSupportSize: number
  // Length of the value representation tensors (number of bins)
  protected readonly valueSupportSize: number
  // Size of hidden layer
  public readonly hiddenLayerSize: number

  // Value loss scale
  private readonly valueScale: number

//  private forwardModel: tf.LayersModel
//  private recurrentModel: tf.LayersModel

  private representationModel: tf.LayersModel
  private predictionModelP: tf.LayersModel
  private predictionModelV: tf.LayersModel
  private dynamicsModelS: tf.LayersModel
  private dynamicsModelR: tf.LayersModel

  private readonly logDir: string

  // Representation network: h(obs)->state
  protected abstract h (observationInput: tf.SymbolicTensor): { s: tf.SymbolicTensor }
  // Prediction network: f(state)->policy,value
  protected abstract f (stateInput: tf.SymbolicTensor): { v: tf.SymbolicTensor, p: tf.SymbolicTensor }
  // Dynamics network: g(state,action)->state,reward
  protected abstract g (actionPlaneInput: tf.SymbolicTensor): { s: tf.SymbolicTensor, r: tf.SymbolicTensor }

  constructor (
    inputSize: number,
    actionSpace: number
  ) {
    // hidden state size
    this.hxSize = 32
    this.rewardSupportSize = 10
    this.valueSupportSize = 10
    this.hiddenLayerSize = 64
    this.valueScale = 0.25

    this.actionSpaceN = actionSpace

    this.logDir = './logs/20230109-005200' // + sysdatetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    // make model for h(o)->s f(s)->p,v
    // s: batch_size x time x state_x x state_y
    const observationInput = tf.input({ shape: [inputSize], name: 'observation_input' })
    const h = this.h(observationInput)
    this.representationModel = tf.model({
      name: 'Representation Model',
      inputs: observationInput,
      outputs: h.s
    })
    this.representationModel.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })

    const hiddenStateInput = tf.input({ shape: [this.hxSize], name: 'hidden_state_input' })
    const f = this.f(hiddenStateInput)
    this.predictionModelP = tf.model({
      name: 'Prediction Policy Model',
      inputs: hiddenStateInput,
      outputs: f.p
    })
    this.predictionModelP.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })
    this.predictionModelV = tf.model({
      name: 'Prediction Value Model',
      inputs: hiddenStateInput,
      outputs: f.v
    })
    this.predictionModelV.compile({
      optimizer: 'sgd',
      loss: tf.losses.sigmoidCrossEntropy
    })

    /*
    const f1 = this.f(h.s)
    const iiModel = tf.model({
      name: 'Initial Inference',
      inputs: observationInput,
      outputs: [h.s, f1.p, f1.v]
    })
    iiModel.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })
    this.forwardModel = iiModel
    */

    // make model for g(s,a)->s,r f(s)->p,v
    // a: one hot encoded vector of shape batch_size x (state_x * state_y)
    const actionPlaneInput = tf.input({ shape: [this.hxSize + this.actionSpaceN], name: 'action_plane_input' })
    const g = this.g(actionPlaneInput)
    this.dynamicsModelS = tf.model({
      name: 'Dynamics State Model',
      inputs: actionPlaneInput,
      outputs: g.s
    })
    this.dynamicsModelS.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy'
    })
    this.dynamicsModelR = tf.model({
      name: 'Dynamics Reward Model',
      inputs: actionPlaneInput,
      outputs: g.r
    })
    this.dynamicsModelR.compile({
      optimizer: 'sgd',
      loss: tf.losses.sigmoidCrossEntropy
    })
    /*
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
    */
  }

  /**
   * initialInference
   * Execute h(o)->s f(s)->p,v
   * @param obs
   */
  public initialInference (obs: tf.Tensor): NetworkOutput {
    const tfHiddenState = this.representationModel.predict(obs.reshape([1, -1])) as tf.Tensor
    const tfPolicy = this.predictionModelP.predict(tfHiddenState) as tf.Tensor
    const tfValue = this.predictionModelV.predict(tfHiddenState) as tf.Tensor
    const hiddenState: number[] = tfHiddenState.reshape([-1]).arraySync() as number[]
    const tfReward = this.rewardTransform(0)
    const reward = 0
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, tfHiddenState, hiddenState)
    /*
    const result = this.forwardModel.predict(obs.reshape([1, -1])) as tf.Tensor[]
    const tfValue = result[2]
    const value = this.inverseValueTransform(result[2])
    const tfReward = this.rewardTransform(0)
    const reward = 0
    const tfPolicy = result[1]
    const policy = this.inversePolicyTransform(result[1])
    const state = result[0]
    const hiddenState: number[] = result[0].reshape([-1]).arraySync() as number[]
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, state, hiddenState)
    */
  }

  /**
   * recurrentInference
   * Execute g(s,a)->s,r f(s)->p,v
   * @param hiddenState
   * @param action
   */
  public recurrentInference (hiddenState: tf.Tensor, action: tf.Tensor): NetworkOutput {
    const x = tf.concat([hiddenState, action], 1)
    const tfNewHiddenState = this.dynamicsModelS.predict(x) as tf.Tensor
    const tfReward = this.dynamicsModelR.predict(x) as tf.Tensor
    const tfPolicy = this.predictionModelP.predict(tfNewHiddenState) as tf.Tensor
    const tfValue = this.predictionModelV.predict(tfNewHiddenState) as tf.Tensor
    const newHiddenState: number[] = tfNewHiddenState.reshape([-1]).arraySync() as number[]
    const reward = this.inverseRewardTransform(tfReward)
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, tfNewHiddenState, newHiddenState)
    /*
    const x = tf.concat([hiddenState, action], 1)
    const result = this.recurrentModel.predict(x) as tf.Tensor[]
    const tfValue = result[3]
    const value = this.inverseValueTransform(result[3])
    const tfReward = result[1]
    const reward = this.inverseRewardTransform(result[1])
    const tfPolicy = result[2]
    const policy = this.inversePolicyTransform(result[2])
    const state = result[0]
    const aHiddenState: number[] = result[0].reshape([-1]).arraySync() as number[]
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, state, aHiddenState)
    */
  }

  public async trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number> {
    const losses: number[] = []
    // Build initial inference
    const observations = tf.concat(samples.map(batch => tf.tensor2d(batch.image).reshape([1, -1])), 0)
    const initialTargets = samples.map(batch => batch.targets[0])
    const targetPolicies = tf.concat(initialTargets.map(target => this.policyPredict(target.policy)), 0)
    const targetValues = tf.concat(initialTargets.map(target => this.valueTransform(target.value)), 0)
    const states = this.representationModel.predict(observations) as tf.Tensor
    const historyP = await this.predictionModelP.fit(states, targetPolicies, { batchSize: 32, epochs: 1, verbose: 0 })
    const historyV = await this.predictionModelV.fit(states, targetValues, { batchSize: 32, epochs: 1, verbose: 0 })
    const lossP = historyP.history["loss"][0] as number
    const lossV = historyP.history["loss"][0] as number
    losses.push(lossP, lossV)
    observations.dispose()
    targetPolicies.dispose()
    targetValues.dispose()
    // Build recurrent inference
    const dataSets = samples.map((batch, index) => {
      const hiddenStates: tf.Tensor[] = []
      const hiddenState = states.gather([index])
      const creps = tf.concat(batch.actions.map((action) => {
        const crep = tf.concat([hiddenStates.at(-1) ?? hiddenState, this.policyTransform(action.id)], 1)
        hiddenStates.push(this.dynamicsModelS.predict(crep) as tf.Tensor)
        return crep
      }), 0)
      const targets = batch.targets.slice(1).filter(target => target.policy.length > 1)
      const targetPolicies = tf.concat(targets.map(target => this.policyPredict(target.policy)), 0)
      const targetValues = tf.concat(targets.map(target => this.valueTransform(target.value)), 0)
      const targetRewards = tf.concat(targets.map(target => this.valueTransform(target.reward)), 0)
      return {
        condRep: creps,
        states: tf.concat(hiddenStates, 0),
        targetPolicies: targetPolicies,
        targetValues: targetValues,
        targetRewards: targetRewards
      }
    })
    for (const dataSet of dataSets) {
      const historyP = await this.predictionModelP.fit(dataSet.states, dataSet.targetPolicies, { batchSize: 32, epochs: 1, verbose: 0 })
      const historyV = await this.predictionModelV.fit(dataSet.states, dataSet.targetValues, { batchSize: 32, epochs: 1, verbose: 0 })
      const historyR = await this.dynamicsModelR.fit(dataSet.condRep, dataSet.targetRewards, { batchSize: 32, epochs: 1, verbose: 0 })
      const lossP = historyP.history["loss"][0] as number
      const lossV = historyP.history["loss"][0] as number
      const lossR = historyR.history["loss"][0] as number
      losses.push(lossP, lossV, lossR)
      dataSet.condRep.dispose()
      dataSet.states.dispose()
      dataSet.targetPolicies.dispose()
      dataSet.targetValues.dispose()
      dataSet.targetRewards.dispose()
    }
    const meanLoss = tf.mean(tf.tensor1d(losses))
    const loss = meanLoss.bufferSync().get(0)
    meanLoss.dispose()
    return loss
  }

  /**
   * trainInitialInference
   * @param obs
   * @param targetState
   * @param targetPolicy
   * @param targetValue
   *
  public async trainInitialInference (observations: number[][][], targets: MuZeroTarget[]): Promise<tf.Tensor[]> {
    const observations = batchSamples.map(batch => batch.image)
    const initialTargets = batchSamples.map(batch => batch.targets[0])
    const obs = observations.map(image => tf.tensor2d(image))
    const targetPolicies = targets.map(target => this.policyPredict(target.policy))
    const targetValues = targets.map(target => this.valueTransform(target.value))
    const states = this.representationModel.predict(obs) as tf.Tensor[]
    const historyP = await this.predictionModelP.fit(states, targetPolicies, { batchSize: 4, epochs: 3 })
    const historyV = await this.predictionModelV.fit(states, targetValues, { batchSize: 4, epochs: 3 })
    debug(`historyP: ${JSON.stringify(historyP.history)}`)
    debug(`historyV: ${JSON.stringify(historyV.history)}`)
    /*
    let state: tf.Tensor = tf.tensor1d(new Array(this.hxSize))
    const policyGradients = tf.variableGrads(() => {
      const result = this.forwardModel.predict(obs.reshape([1, -1])) as tf.Tensor[]
      state = tf.keep(result[0])
      return this.lossPolicy(targetPolicy, result[1]).add(this.lossValue(targetValue, result[2]))
    })

      historyP.history.loss.reduce((s ,v) => s+(v as number), 0) +
      historyV.history.loss.reduce((s ,v) => s+(v as number), 0),
    *
    return states
  }

  /**
   * trainRecurrentInference
   * @param hiddenState
   * @param action
   * @param targetState
   * @param targetPolicy
   * @param targetValue
   * @param targetReward
   * @param lossScale
   *
  public async trainRecurrentInference (states: tf.Tensor[], targets: { action: tf.Tensor, target: MuZeroTarget }[][], lossScale: number): Promise<number> {
//    this.forwardModel.summary()
    const x = tf.concat([states, action], 1)
    const newStates = this.dynamicsModelS.predict(x)
    const historyR = await this.dynamicsModelR.fit(x, this.valueTransform(targetReward), { batchSize: 1, epochs: 3 })
    const targetPolicies = targets.map(target => this.policyPredict(target.policy))
    const targetValues = targets.map(target => this.valueTransform(target.value))
    const historyP = await this.predictionModelP.fit(states, targetPolicies, { batchSize: 4, epochs: 3 })
    const historyV = await this.predictionModelV.fit(states, targetValues, { batchSize: 4, epochs: 3 })
    debug(`historyP: ${JSON.stringify(historyP.history)}`)
    debug(`historyV: ${JSON.stringify(historyV.history)}`)


    /*
    let state: tf.Tensor = tf.tensor1d(new Array(this.hxSize))
    const policyGradients = tf.variableGrads(() => {
      const result = this.recurrentModel.predict(x) as tf.Tensor[]
      state = tf.keep(result[0])
      return this.lossReward(targetReward, result[1]).add(this.lossPolicy(targetPolicy, result[2])).add(this.lossValue(targetValue, result[3])).mul(lossScale)
    })
    return {
      grads: policyGradients.grads,
      loss: policyGradients.value,
      state
    }
    *
  }
  */
  public lossReward (targetReward: number, result: tf.Tensor): tf.Scalar {
    return tf.losses.sigmoidCrossEntropy(this.rewardTransform(targetReward), result).asScalar()
  }

  public lossValue (targetValue: number, result: tf.Tensor): tf.Scalar {
    return tf.losses.sigmoidCrossEntropy(this.valueTransform(targetValue), result).asScalar().mul(this.valueScale)
  }

  public lossPolicy (targetPolicy: number[], result: tf.Tensor): tf.Scalar {
    return tf.losses.softmaxCrossEntropy(this.policyPredict(targetPolicy), result).asScalar()
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
      this.representationModel.save(path + '_representation'),
      this.predictionModelP.save(path + '_prediction_p'),
      this.predictionModelV.save(path + '_prediction_v'),
      this.dynamicsModelS.save(path + '_dynamics_s'),
      this.dynamicsModelR.save(path + '_dynamics_r')
    ])
  }

  public async load (path: string): Promise<void> {
    try {
      const [
        representationModel,
        predictionModelP,
        predictionModelV,
        dynamicsModelS,
        dynamicsModelR
      ] = await Promise.all([
        tf.loadLayersModel(path + '_representation/model.json'),
        tf.loadLayersModel(path + '_prediction_p/model.json'),
        tf.loadLayersModel(path + '_prediction_v/model.json'),
        tf.loadLayersModel(path + '_dynamics_s/model.json'),
        tf.loadLayersModel(path + '_dynamics_r/model.json')
      ])
      debug(`Disposed ${this.dispose()} tensors`)
      this.representationModel = representationModel
      this.predictionModelP = predictionModelP
      this.predictionModelV = predictionModelV
      this.dynamicsModelS = dynamicsModelS
      this.dynamicsModelR = dynamicsModelR
    } catch (e) {
      throw e
    }
  }

  public copyWeights (network: BaseMuZeroNet): void {
    tf.tidy(() => {
      network.representationModel.setWeights(this.representationModel.getWeights())
      network.predictionModelP.setWeights(this.predictionModelP.getWeights())
      network.predictionModelV.setWeights(this.predictionModelV.getWeights())
      network.dynamicsModelS.setWeights(this.dynamicsModelS.getWeights())
      network.dynamicsModelR.setWeights(this.dynamicsModelR.getWeights())
    })
  }

  public dispose (): number {
    let disposed = 0
    disposed += this.representationModel.dispose().numDisposedVariables
    disposed += this.predictionModelP.dispose().numDisposedVariables
    disposed += this.predictionModelV.dispose().numDisposedVariables
    disposed += this.dynamicsModelS.dispose().numDisposedVariables
    disposed += this.dynamicsModelR.dispose().numDisposedVariables
    return disposed
  }

  public trainingSteps (): number {
    return 1
  }
}
