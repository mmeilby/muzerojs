import * as tf from '@tensorflow/tfjs-node'
import { scalarToSupport, supportToScalar } from '../selfplay/utils'
import { NetworkOutput } from './networkoutput'
import debugFactory from 'debug'
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise} from "../selfplay/entities";
import {Logs} from "@tensorflow/tfjs-node";

const debug = debugFactory('muzero:network:debug')

export abstract class BaseMuZeroNet {
  // Length of the hidden state tensors (number of outputs for g.s and h.s)
  protected readonly hxSize: number
  // Length of the action tensors
  protected readonly actionSpaceN: number
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
    actionSpace: number,
    learningRate: number
  ) {
    // hidden state size
    this.hxSize = 32
    this.rewardSupportSize = 10
    this.valueSupportSize = 10
    this.hiddenLayerSize = 64
    this.valueScale = 0.25
    this.weightDecay = 0.0001

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
      loss: tf.losses.softmaxCrossEntropy, metrics: ['acc']
    })

    const hiddenStateInput = tf.input({ shape: [this.hxSize], name: 'hidden_state_input' })
    const f = this.f(hiddenStateInput)
    this.predictionModelP = tf.model({
      name: 'Prediction Policy Model',
      inputs: hiddenStateInput,
      outputs: f.p
    })
    this.predictionModelP.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: tf.losses.softmaxCrossEntropy, metrics: ['acc']
    })
    this.predictionModelV = tf.model({
      name: 'Prediction Value Model',
      inputs: hiddenStateInput,
      outputs: f.v
    })
    const scale = this.valueScale
    function valueLoss(multiClassLabels: tf.Tensor, logits: tf.Tensor, weights?: tf.Tensor, labelSmoothing?: number, reduction?: tf.Reduction) {
      return tf.losses.sigmoidCrossEntropy(multiClassLabels, logits, tf.scalar(scale), labelSmoothing, reduction)
    }
    this.predictionModelV.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: valueLoss, metrics: ['acc']
    })

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
      loss: tf.losses.softmaxCrossEntropy, metrics: ['acc']
    })
    this.dynamicsModelR = tf.model({
      name: 'Dynamics Reward Model',
      inputs: actionPlaneInput,
      outputs: g.r
    })
    this.dynamicsModelR.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: tf.losses.sigmoidCrossEntropy, metrics: ['acc']
    })
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
  }

  /**
   * recurrentInference
   * Execute g(s,a)->s,r f(s)->p,v
   * @param hiddenState
   * @param action
   */
  public recurrentInference (hiddenState: tf.Tensor, action: tf.Tensor): NetworkOutput {
    const conditionedHiddenState = tf.concat([hiddenState, action], 1)
    const tfNewHiddenState = this.dynamicsModelS.predict(conditionedHiddenState) as tf.Tensor
    const tfReward = this.dynamicsModelR.predict(conditionedHiddenState) as tf.Tensor
    const tfPolicy = this.predictionModelP.predict(tfNewHiddenState) as tf.Tensor
    const tfValue = this.predictionModelV.predict(tfNewHiddenState) as tf.Tensor
    const newHiddenState: number[] = tfNewHiddenState.reshape([-1]).arraySync() as number[]
    const reward = this.inverseRewardTransform(tfReward)
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(tfValue, value, tfReward, reward, tfPolicy, policy, tfNewHiddenState, newHiddenState)
  }

  /**
   * trainInference
   * @param samples
   */
  public async trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number[]> {
    const losses: number[] = []
    const accuracy: number[] = []
    const fittingArgs = {
      batchSize: 32,
      epochs: 4,
      verbose: 0,
      shuffle: true,
/*
      callbacks: {
        onEpochEnd: async (epoch: number, logs: Logs | undefined): Promise<void> => {
          debug(`${epoch}: loss=${JSON.stringify(logs?.loss)} acc=${JSON.stringify(logs?.acc)}`)
        }
      }
*/
    }

    async function trackLoss(phistory: Promise<tf.History>) {
      const history = await phistory
      losses.push(history.history["loss"][0] as number)
      accuracy.push(history.history["acc"][0] as number)
    }
    // Build initial inference
    const observations = tf.tidy(() => tf.concat(samples.map(batch => tf.tensor2d(batch.image).reshape([1, -1])), 0))
    const initialTargets = samples.map(batch => batch.targets[0])
    const targetPolicies = tf.tidy(() => tf.concat(initialTargets.map(target => this.policyPredict(target.policy)), 0))
    const targetValues = tf.tidy(() => tf.concat(initialTargets.map(target => this.valueTransform(target.value)), 0))
    const states = this.representationModel.predict(observations) as tf.Tensor
    await trackLoss(this.predictionModelP.fit(states, targetPolicies, fittingArgs))
    await trackLoss(this.predictionModelV.fit(states, targetValues, fittingArgs))
    observations.dispose()
    targetPolicies.dispose()
    targetValues.dispose()

    // Build recurrent inference
    const dataSets = tf.tidy(() => samples.map((batch, index) => {
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
    }))
    states.dispose()
    for (const dataSet of dataSets) {
      await trackLoss(this.predictionModelP.fit(dataSet.states, dataSet.targetPolicies, fittingArgs))
      await trackLoss(this.predictionModelV.fit(dataSet.states, dataSet.targetValues, fittingArgs))
      await trackLoss(this.dynamicsModelR.fit(dataSet.condRep, dataSet.targetRewards, fittingArgs))
      dataSet.condRep.dispose()
      dataSet.states.dispose()
      dataSet.targetPolicies.dispose()
      dataSet.targetValues.dispose()
      dataSet.targetRewards.dispose()
    }
    const meanAccuracy = tf.tidy(() => tf.mean(tf.tensor1d(accuracy)))
    const acc = meanAccuracy.bufferSync().get(0)
    meanAccuracy.dispose()
    const meanLoss = tf.tidy(() => tf.mean(tf.tensor1d(losses)))
    const loss = meanLoss.bufferSync().get(0)
    meanLoss.dispose()
    return [ loss, acc ]
  }

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
      this.representationModel.save(path + 'rep'),
      this.predictionModelP.save(path + 'pre_p'),
      this.predictionModelV.save(path + 'pre_v'),
      this.dynamicsModelS.save(path + 'dyn_s'),
      this.dynamicsModelR.save(path + 'dyn_r')
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
        tf.loadLayersModel(path + 'rep/model.json'),
        tf.loadLayersModel(path + 'pre_p/model.json'),
        tf.loadLayersModel(path + 'pre_v/model.json'),
        tf.loadLayersModel(path + 'dyn_s/model.json'),
        tf.loadLayersModel(path + 'dyn_r/model.json')
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
