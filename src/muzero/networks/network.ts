import * as tf from '@tensorflow/tfjs-node'
import {LogMetrics, scalarToSupport, supportToScalar} from './utils'
import { NetworkOutput } from './networkoutput'
import {MuZeroBatch} from "../replaybuffer/batch";
import {Actionwise} from "../selfplay/entities";
import {Logs} from "@tensorflow/tfjs-node";
import {MuZeroHiddenState, MuZeroNetwork, MuZeroObservation} from "./nnet";
import debugFactory from 'debug'
import {MuZeroAction} from "../games/core/action";

const debug = debugFactory('muzero:network:debug')

export class MuZeroNetObservation implements MuZeroObservation {
  constructor(
      public state: number[][]
  ) {}
}

class MuZeroNetHiddenState implements MuZeroHiddenState {
  constructor(
      public state: number[]
  ) {}
}

export class MuZeroNet implements MuZeroNetwork<MuZeroAction> {
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

  private initialInferenceModel: tf.LayersModel
  private recurrentInferenceModel: tf.LayersModel

  private readonly logDir: string

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
    const observationInput = tf.input({ shape: [inputSize], name: 'observation_input' })
    const h = this.h()
    const f = this.f()
    const scale = this.valueScale
    function valueLoss(multiClassLabels: tf.Tensor, logits: tf.Tensor, weights?: tf.Tensor, labelSmoothing?: number, reduction?: tf.Reduction) {
      return tf.losses.sigmoidCrossEntropy(multiClassLabels, logits, tf.scalar(scale), labelSmoothing, reduction)
    }

    const representationLayer = h.s.apply(h.sh.apply(observationInput))
    // make model for initial inference f(h(o))->p,v
    this.initialInferenceModel = tf.model({
      name: 'Initial Inference Model',
      inputs: observationInput,
      outputs: [
        f.p.apply(f.ph.apply(representationLayer)) as tf.SymbolicTensor,
        f.v.apply(f.vh.apply(representationLayer)) as tf.SymbolicTensor,
        representationLayer as tf.SymbolicTensor
      ]
    })
    this.initialInferenceModel.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: {
        prediction_policy_output: tf.losses.softmaxCrossEntropy,
        prediction_value_output: valueLoss,
        representation_state_output: () => tf.scalar(0)
      },
      metrics: ['acc']
    })

    // make model for g(s,a)->s,r f(s)->p,v
    // a: one hot encoded vector of shape batch_size x (state_x * state_y)
    const actionPlaneInput = tf.input({ shape: [this.hxSize + this.actionSpaceN], name: 'action_plane_input' })
    const g = this.g()
    const dynamicsLayer = g.s.apply(g.sh.apply(actionPlaneInput))
    // make model for recurrent inference f(g(s,a))->p,v,r
    this.recurrentInferenceModel = tf.model({
      name: 'Recurrent Inference Model',
      inputs: actionPlaneInput,
      outputs: [
        f.p.apply(f.ph.apply(dynamicsLayer)) as tf.SymbolicTensor,
        f.v.apply(f.vh.apply(dynamicsLayer)) as tf.SymbolicTensor,
        g.r.apply(g.rh.apply(actionPlaneInput)) as tf.SymbolicTensor,
        dynamicsLayer as tf.SymbolicTensor
      ]
    })
    this.recurrentInferenceModel.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: {
        prediction_policy_output: tf.losses.softmaxCrossEntropy,
        prediction_value_output: valueLoss,
        dynamics_reward_output: valueLoss,
        dynamics_state_output: () => tf.scalar(0)
      },
      metrics: ['acc']
    })
  }

  private makeHiddenLayer (name: string, units: number): tf.layers.Layer {
    return tf.layers.dense({
      name: name,
      units: units,
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
      activation: 'softsign',
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
      units: this.valueSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    const fp = tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.actionSpaceN,
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

  // Dynamics network: g(state,action)->state,reward
  private g (): { sh: tf.layers.Layer, s: tf.layers.Layer, rh: tf.layers.Layer, r: tf.layers.Layer } {
    const gs = tf.layers.dense({
      name: 'dynamics_state_output',
      units: this.hxSize,
      activation: 'softsign',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    const gr = tf.layers.dense({
      name: 'dynamics_reward_output',
      units: this.rewardSupportSize * 2 + 1,
      activation: 'linear', // softmax?
      kernelInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    })
    return {
      sh: this.makeHiddenLayer('dynamics_state_hidden', this.hiddenLayerSize),
      s: gs,
      rh: this.makeHiddenLayer('dynamics_reward_hidden', this.hiddenLayerSize),
      r: gr
    }
  }

  /**
   * initialInference
   * Execute h(o)->s f(s)->p,v
   * @param obs
   */
  public initialInference (obs: MuZeroNetObservation): NetworkOutput {
    const observation = tf.tensor2d(obs.state)
    const [ tfPolicy, tfValue, tfHiddenState ] = this.initialInferenceModel.predict(observation.reshape([1, -1])) as tf.Tensor[]
    const hiddenState: MuZeroNetHiddenState = new MuZeroNetHiddenState(tfHiddenState.reshape([-1]).arraySync() as number[])
    const reward = 0
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(value, reward, policy, hiddenState)
  }

  /**
   * recurrentInference
   * Execute g(s,a)->s,r f(s)->p,v
   * @param hiddenState
   * @param action
   */
  public recurrentInference (hiddenState: MuZeroNetHiddenState, action: MuZeroAction): NetworkOutput {
    const conditionedHiddenState = tf.concat([tf.tensor2d([hiddenState.state]), this.policyTransform(action.id)], 1)
    const [ tfPolicy, tfValue, tfReward, tfNewHiddenState ] = this.recurrentInferenceModel.predict(conditionedHiddenState) as tf.Tensor[]
    const newHiddenState: MuZeroNetHiddenState = new MuZeroNetHiddenState(tfNewHiddenState.reshape([-1]).arraySync() as number[])
    const reward = this.inverseRewardTransform(tfReward)
    const policy = this.inversePolicyTransform(tfPolicy)
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(value, reward, policy, newHiddenState)
  }

  /**
   * trainInference
   * @param samples
   */
  public async trainInference (samples: MuZeroBatch<Actionwise>[]): Promise<number[]> {
    const masterLog: LogMetrics[] = []
//    const metrics: Map<string, LogMetrics[]> = new Map<string, LogMetrics[]>([
//        ['policy', []], ['value', []], ['reward', []]
//    ])
    const onBatchEnd = async (batch: number, logs?: Logs): Promise<void> => {
      /*
      const meanPolicy = tf.tidy(() => tf.mean(tf.tensor1d((logs["prediction_policy_output_loss"] ?? [0]) as number[])))
      const meanValue = tf.tidy(() => tf.mean(tf.tensor1d((history.history["prediction_value_output_loss"] ?? [0]) as number[])))
      const meanReward = tf.tidy(() => tf.mean(tf.tensor1d((history.history["dynamics_reward_output_loss"] ?? [0]) as number[])))
      debug(`Loss metrics: policy=${meanPolicy.bufferSync().get(0).toFixed(3)}, value=${meanValue.bufferSync().get(0).toFixed(3)}, reward=${meanReward.bufferSync().get(0).toFixed(3)}`)
      meanPolicy.dispose()
      meanValue.dispose()
      meanReward.dispose()
      const meanloss = tf.tidy(() => tf.mean(tf.tensor1d((history.history["loss"] ?? [0]) as number[])).bufferSync().get(0))
      const meanAcc = tf.tidy(() => tf.mean(tf.tensor1d(
          ((history.history["prediction_policy_output_acc"] ?? [0]) as number[])
              .concat((history.history["prediction_value_output_acc"] ?? [0]) as number[])
              .concat((history.history["dynamics_reward_output_acc"] ?? [0]) as number[])
      )).bufferSync().get(0))
      return new LogMetrics(meanloss, meanAcc)

       */
      const meanAcc = tf.tidy(() => tf.mean(tf.tensor1d([
        logs?.prediction_policy_output_acc ?? 1,
        logs?.prediction_value_output_acc ?? 1,
        logs?.dynamics_reward_output_acc ?? 1,
//            logs?.dynamics_state_output_acc ?? 1,
//            logs?.representation_state_output_acc ?? 1,
      ])).bufferSync().get(0))
      masterLog.push(new LogMetrics(
          logs?.loss ?? 0,
          meanAcc,
          logs?.prediction_policy_output_loss ?? 0,
          logs?.prediction_policy_output_acc ?? 1,
          logs?.prediction_value_output_loss ?? 0,
          logs?.prediction_value_output_acc ?? 1,
          logs?.dynamics_reward_output_loss ?? 0,
          logs?.dynamics_reward_output_acc ?? 1))
    }
    /*
    const fittingArgs = {
      batchSize: 32,
      epochs: 4,
      verbose: 0,
      shuffle: true,

      callbacks: {
        onBatchEnd: onBatchEnd
      }

    }
    */
    debug(`Training initial batch of size=${samples.length}`)
    // Build initial inference
    const observations = tf.tidy(() => tf.concat(samples.map(batch => tf.tensor2d((batch.image as MuZeroNetObservation).state).reshape([1, -1])), 0))
    const initialTargets = samples.map(batch => batch.targets[0])
    const targetPolicies = tf.tidy(() => tf.concat(initialTargets.map(target => this.policyPredict(target.policy)), 0))
    const targetValues = tf.tidy(() => tf.concat(initialTargets.map(target => this.valueTransform(target.value)), 0))
    const states = tf.tidy(() => (this.initialInferenceModel.predict(observations) as tf.Tensor[])[2])
    await this.initialInferenceModel.fit(
        observations,
        {
          prediction_policy_output: targetPolicies,
          prediction_value_output: targetValues,
          representation_state_output: states
        },
        {
          batchSize: 16,
          epochs: 1,
          verbose: 0,
          shuffle: true,
          callbacks: { onBatchEnd: onBatchEnd }
        }
    )
    observations.dispose()
    targetPolicies.dispose()
    targetValues.dispose()
    // Build recurrent inference
    const dataSets = tf.tidy(() => samples.map((batch, index) => {
      const hiddenStates: tf.Tensor[] = []
      const hiddenState = states.gather([index])
      const creps = tf.concat(batch.actions.map((action) => {
        const crep = tf.concat([hiddenStates.at(-1) ?? hiddenState, this.policyTransform(action.id)], 1)
        hiddenStates.push((this.recurrentInferenceModel.predict(crep) as tf.Tensor[])[3])
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
    const condReps = tf.concat(dataSets.map(dataSet => dataSet.condRep))
    const rTargetPolicies = tf.concat(dataSets.map(dataSet => dataSet.targetPolicies))
    const rTargetValues = tf.concat(dataSets.map(dataSet => dataSet.targetValues))
    const rTargetRewards = tf.concat(dataSets.map(dataSet => dataSet.targetRewards))
    const rStates = tf.concat(dataSets.map(dataSet => dataSet.states))
    for (const dataSet of dataSets) {
      dataSet.condRep.dispose()
      dataSet.states.dispose()
      dataSet.targetPolicies.dispose()
      dataSet.targetValues.dispose()
      dataSet.targetRewards.dispose()
    }
    debug(`Training recurrent batch of size=${condReps.shape[0]}`)
    await this.recurrentInferenceModel.fit(
        condReps,
        {
          prediction_policy_output: rTargetPolicies,
          prediction_value_output: rTargetValues,
          dynamics_reward_output: rTargetRewards,
          dynamics_state_output: rStates
        },
        {
          batchSize: 16,
          epochs: 1,
          verbose: 0,
          shuffle: true,
          callbacks: { onBatchEnd: onBatchEnd }
        }
    )
    condReps.dispose()
    rTargetPolicies.dispose()
    rTargetValues.dispose()
    rTargetRewards.dispose()
    rStates.dispose()
    const acc = tf.tidy(() => tf.mean(tf.tensor1d(masterLog.map(lm => lm.accuracy))).bufferSync().get(0))
    const loss = tf.tidy(() => tf.mean(tf.tensor1d(masterLog.map(lm => lm.loss))).bufferSync().get(0))
    return [ loss, acc ]
  }
/*
  private lossReward (targetReward: number, result: tf.Tensor): tf.Scalar {
    return tf.losses.sigmoidCrossEntropy(this.rewardTransform(targetReward), result).asScalar()
  }

  private lossValue (targetValue: number, result: tf.Tensor): tf.Scalar {
    return tf.losses.sigmoidCrossEntropy(this.valueTransform(targetValue), result).asScalar().mul(this.valueScale)
  }

  private lossPolicy (targetPolicy: number[], result: tf.Tensor): tf.Scalar {
    return tf.losses.softmaxCrossEntropy(this.policyPredict(targetPolicy), result).asScalar()
  }
*/
  private inversePolicyTransform (x: tf.Tensor): number[] {
    return x.squeeze().arraySync() as number[]
  }

  private policyTransform (policy: number): tf.Tensor {
    // One hot encode integer actions to Tensor2D
    return tf.oneHot(tf.tensor1d([policy], 'int32'), this.actionSpaceN, 1, 0, 'float32')
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
      this.initialInferenceModel.save(path + 'ii'),
      this.recurrentInferenceModel.save(path + 'ri'),
    ])
  }

  public async load (path: string): Promise<void> {
    try {
      const [
        initialInference,
        recurrentInference,
      ] = await Promise.all([
        tf.loadLayersModel(path + 'ii/model.json'),
        tf.loadLayersModel(path + 'ri/model.json'),
      ])
      this.initialInferenceModel.setWeights(initialInference.getWeights())
      this.recurrentInferenceModel.setWeights(recurrentInference.getWeights())
      initialInference.dispose()
      recurrentInference.dispose()
    } catch (e) {
      throw e
    }
  }

  public copyWeights (network: MuZeroNet): void {
    tf.tidy(() => {
      network.initialInferenceModel.setWeights(this.initialInferenceModel.getWeights())
      network.recurrentInferenceModel.setWeights(this.recurrentInferenceModel.getWeights())
    })
  }
/*
  private dispose (): number {
    let disposed = 0
    disposed += this.initialInferenceModel.dispose().numDisposedVariables
    disposed += this.recurrentInferenceModel.dispose().numDisposedVariables
    return disposed
  }

  private trainingSteps (): number {
    return 1
  }
*/
}
