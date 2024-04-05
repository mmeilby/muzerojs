import * as tf from '@tensorflow/tfjs-node'
import { type Scalar, type Tensor, tensor } from '@tensorflow/tfjs-node'
import { scalarToSupport, supportToScalar } from './utils'
import { NetworkOutput } from './networkoutput'
import { type MuZeroBatch } from '../replaybuffer/batch'
import { type Actionwise } from '../selfplay/entities'
import { type MuZeroHiddenState, type MuZeroNetwork, type MuZeroObservation } from './nnet'
import debugFactory from 'debug'
import { type MuZeroTarget } from '../replaybuffer/target'

const debug = debugFactory('muzero:network:debug')

export class MuZeroNetObservation implements MuZeroObservation {
  constructor (
    public observation: number[]
  ) {}
}

class MuZeroNetHiddenState implements MuZeroHiddenState {
  constructor (
    public state: number[]
  ) {}
}

class Prediction {
  constructor (
    public scale: number,
    public value: Tensor,
    public reward: Tensor,
    public policy: Tensor
  ) {}
}

class LossLog {
  public value: number
  public reward: number
  public policy: number
  public total: tf.Tensor

  constructor () {
    this.value = 0
    this.reward = 0
    this.policy = 0
    this.total = tf.scalar(0)
  }
}

export class MuZeroNet<Action extends Actionwise> implements MuZeroNetwork<Action> {
  // Length of the hidden state tensors (number of outputs for g.s and h.s)
  protected readonly hxSize: number
  // Length of the reward representation tensors (number of bins)
  protected readonly rewardSupportSize: number
  // Length of the value representation tensors (number of bins)
  protected readonly valueSupportSize: number
  // Size of hidden layer
  public readonly hiddenLayerSize: number[]

  // Scale the value loss to avoid over fitting of the value function,
  // paper recommends 0.25 (See paper appendix Reanalyze)
  private readonly valueScale: number
  // L2 weights regularization
  protected readonly weightDecay: number

  private representationModel: tf.Sequential = tf.sequential()
  private valueModel: tf.Sequential = tf.sequential()
  private policyModel: tf.Sequential = tf.sequential()
  private dynamicsModel: tf.Sequential = tf.sequential()
  private rewardModel: tf.Sequential = tf.sequential()

  private readonly logDir: string

  constructor (
    private readonly inputSize: number,
    // Length of the action tensors
    private readonly actionSpaceN: number,
    // Learning rate for SGD
    private readonly learningRate: number,
    hiddenLayerSizes: number | number[] = 32
  ) {
    // hidden state size
    this.hxSize = 32
    this.rewardSupportSize = 0
    this.valueSupportSize = 0
    this.hiddenLayerSize = !Array.isArray(hiddenLayerSizes) ? [hiddenLayerSizes] : hiddenLayerSizes
    this.valueScale = 0.25
    this.weightDecay = 0.0001

    this.logDir = './logs/20230109-005200' // + sysdatetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    this.makeModels()
  }

  private makeModels (): void {
    const repModel = tf.sequential()
    this.makeHiddenLayer(repModel, 'representation_state_hidden', [this.inputSize])
    repModel.add(tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'linear',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.representationModel = repModel
    const valueModel = tf.sequential()
    this.makeHiddenLayer(valueModel, 'prediction_value_hidden', [this.hxSize])
    valueModel.add(tf.layers.dense({
      name: 'prediction_value_output',
      units: 1,
      activation: 'relu',
//      activation: 'tanh',
//      kernelInitializer: 'zeros',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.valueModel = valueModel
    const policyModel = tf.sequential()
    this.makeHiddenLayer(policyModel, 'prediction_policy_hidden', [this.hxSize])
    policyModel.add(tf.layers.dense({
      name: 'prediction_policy_output',
      units: this.actionSpaceN,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.policyModel = policyModel
    const rewardModel = tf.sequential()
    this.makeHiddenLayer(rewardModel, 'dynamics_reward_hidden', [this.hxSize + this.actionSpaceN])
    rewardModel.add(tf.layers.dense({
      name: 'dynamics_reward_output',
      units: 1,
      activation: 'relu',
//      activation: 'tanh',
//      kernelInitializer: 'zeros',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.rewardModel = rewardModel
    const dynamicsModel = tf.sequential()
    this.makeHiddenLayer(dynamicsModel, 'dynamics_state_hidden', [this.hxSize + this.actionSpaceN])
    dynamicsModel.add(tf.layers.dense({
      name: 'dynamics_state_output',
      units: this.hxSize,
      activation: 'linear',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.dynamicsModel = dynamicsModel
  }

  private makeHiddenLayer (model: tf.Sequential, name: string, inputShape: number[]): void {
    this.hiddenLayerSize.forEach((units, i) => {
      model.add(tf.layers.dense({
        name,
        // `inputShape` is required only for the first layer.
        inputShape: i === 0 ? inputShape : undefined,
        units,
        activation: 'relu',
        kernelInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
        useBias: false
      }))
    })
  }

  /**
   * initialInference
   * Execute h(o)->s f(s)->p,v
   * @param obs
   */
  public initialInference (obs: MuZeroObservation): NetworkOutput {
    if (!(obs instanceof MuZeroNetObservation)) {
      throw new Error('Incorrect observation applied to initialInference')
    }
    const observation = tf.tensor1d(obs.observation).reshape([1, -1])
    const tfHiddenState = this.representationModel.predict(observation) as tf.Tensor
    const hiddenState: MuZeroNetHiddenState = new MuZeroNetHiddenState(tfHiddenState.reshape([-1]).arraySync() as number[])
    const reward = 0
    const tfPolicy = this.policyModel.predict(tfHiddenState) as tf.Tensor
    const policy = this.inversePolicyTransform(tfPolicy)
    const tfValue = this.valueModel.predict(tfHiddenState) as tf.Tensor
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(value, reward, policy, hiddenState)
  }

  /**
   * recurrentInference
   * Execute g(s,a)->s',r f(s')->p,v
   * @param hiddenState
   * @param action
   */
  public recurrentInference (hiddenState: MuZeroHiddenState, action: Actionwise): NetworkOutput {
    if (!(hiddenState instanceof MuZeroNetHiddenState)) {
      throw new Error('Incorrect hidden state applied to recurrentInference')
    }
    const conditionedHiddenState = tf.concat([tf.tensor2d([hiddenState.state]), this.policyTransform(action.id)], 1)
    const tfHiddenState = this.dynamicsModel.predict(conditionedHiddenState) as tf.Tensor
    const newHiddenState: MuZeroNetHiddenState = new MuZeroNetHiddenState(tfHiddenState.reshape([-1]).arraySync() as number[])
    const tfReward = this.rewardModel.predict(conditionedHiddenState) as tf.Tensor
    const reward = this.inverseRewardTransform(tfReward)
    const tfPolicy = this.policyModel.predict(tfHiddenState) as tf.Tensor
    const policy = this.inversePolicyTransform(tfPolicy)
    const tfValue = this.valueModel.predict(tfHiddenState) as tf.Tensor
    const value = this.inverseValueTransform(tfValue)
    return new NetworkOutput(value, reward, policy, newHiddenState)
  }

  public trainInference (samples: Array<MuZeroBatch<Actionwise>>): number[] {
    debug(`Training sample set of ${samples.length} games`)
    const f = () => {
      const batchLosses: LossLog = new LossLog()
      for (const batch of samples) {
        const predictions = this.calculatePredictions(batch)
        const lossAndGradients = this.measureLoss(predictions, batch.targets)
        batchLosses.total = batchLosses.total.add(lossAndGradients.total)
        if (debug.enabled) {
          const lossV = lossAndGradients.value.toFixed(3)
          const lossR = lossAndGradients.reward.toFixed(3)
          const lossP = lossAndGradients.policy.toFixed(3)
          const total = lossAndGradients.total.bufferSync().get(0).toFixed(3)
          debug(`Game overall loss: V=${lossV}, R=${lossR}, P=${lossP} T=${total}`)
        }
        lossAndGradients.total.dispose()
      }
      batchLosses.total = batchLosses.total.div(samples.length)
      if (debug.enabled) {
        debug(`Sample set mean loss: T=${batchLosses.total.bufferSync().get(0).toFixed(3)}`)
      }
      // update weights
      return batchLosses.total.asScalar()
    }
    const optimizer = tf.train.sgd(this.learningRate)
//    const representationWeights = tf.concat(this.representationModel.getWeights(true))
    const cost = optimizer.minimize(f, true)
//    const representationWeightsAfterTraining = tf.concat(this.representationModel.getWeights(true))
//    if (representationWeightsAfterTraining.notEqual(representationWeights)) {
//      representationWeights.sub(representationWeightsAfterTraining).print()
//    }
    const loss = cost?.bufferSync().get(0) ?? 0
    cost?.dispose()
    optimizer.dispose()
    return [loss, 0]
  }

  public calculatePredictions (batch: MuZeroBatch<Actionwise>): Prediction[] {
    const observation = tf.tensor1d((batch.image as MuZeroNetObservation).observation).reshape([1, -1])
    const tfHiddenState = this.representationModel.predict(observation) as tf.Tensor
    // Gradient scaling controls the representation network training. To prevent training set scale = 0
    let state = this.scaleGradient(tfHiddenState, 1)
    const tfPolicy = this.policyModel.predict(state) as tf.Tensor
    const tfValue = this.valueModel.predict(state) as tf.Tensor
    const predictions: Prediction[] = [{
      scale: 1,
      value: tfValue,
      reward: tensor(0),
      policy: tfPolicy
    }]
    for (const action of batch.actions) {
      const conditionedHiddenState = tf.concat([state, this.policyTransform(action.id)], 1)
      const tfNewHiddenState = this.dynamicsModel.predict(conditionedHiddenState) as tf.Tensor
      const tfReward = this.rewardModel.predict(conditionedHiddenState) as tf.Tensor
      const tfPolicy = this.policyModel.predict(tfNewHiddenState) as tf.Tensor
      const tfValue = this.valueModel.predict(tfNewHiddenState) as tf.Tensor
      predictions.push({
        scale: 1 / batch.actions.length,
        value: tfValue,
        reward: tfReward,
        policy: tfPolicy
      })
      // Prepare new state for next game step
      // Gradient scaling controls the dynamics network training. To prevent training set scale = 0
      state = this.scaleGradient(tfNewHiddenState, 0.5)
    }
    return predictions
  }

  public measureLoss (predictions: Prediction[], targets: MuZeroTarget[]): LossLog {
    const batchTotalLoss: LossLog = new LossLog()
    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i]
      const target = targets[i]
      const lossV = this.valueLoss(prediction.value, this.valueTransform(target.value))
      batchTotalLoss.value += lossV.bufferSync().get(0)
      batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossV, prediction.scale))
      if (i > 0) {
        const lossR = this.scalarLoss(prediction.reward, this.valueTransform(target.reward))
        batchTotalLoss.reward += lossR.bufferSync().get(0)
        batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossR, prediction.scale))
      }
      if (target.policy.length > 0) {
        const lossP = tf.losses.softmaxCrossEntropy(this.policyPredict(target.policy), prediction.policy).asScalar()
        batchTotalLoss.policy += lossP.bufferSync().get(0)
        batchTotalLoss.total = batchTotalLoss.total.add(this.scaleGradient(lossP, prediction.scale))
      }
    }
    batchTotalLoss.value /= predictions.length
    batchTotalLoss.reward /= predictions.length
    batchTotalLoss.policy /= predictions.length
    return batchTotalLoss
  }

  private getTrainableVariables (model: tf.LayersModel): tf.Variable[] {
    return model.getWeights(true) as Array<tf.Variable<tf.Rank>>
  }

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
  private pushGradients (record: Record<string, tf.Tensor[]>, gradients: tf.NamedTensorMap): void {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key])
      } else {
        record[key] = [gradients[key]]
      }
    }
  }

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[][]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor[]}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
  private pushGradientArrays (record: Record<string, tf.Tensor[][]>, gradients: Record<string, tf.Tensor[]>): void {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key])
      } else {
        record[key] = [gradients[key]]
      }
    }
  }

  /**
   * Scale the gradient values using normalized reward values and compute average.
   *
   * The gradient values are scaled by the normalized reward values. Then they
   * are averaged across all games and all steps.
   *
   * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
   *   name to all the gradient values for the variable across all games and all
   *   steps.
   * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
   *   for all the games. Each element of the Array is a 1D tf.Tensor of which
   *   the length equals the number of steps in the game.
   * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
   *   for the variables.
   */
  private scaleAndAverageGradients (allGradients: Record<string, tf.Tensor[][]>): tf.NamedTensorMap {
    return tf.tidy(() => {
      const gradients: tf.NamedTensorMap = {}
      for (const varName in allGradients) {
        gradients[varName] = tf.tidy(() => {
          // Stack gradients together.
          const varGradients = allGradients[varName].map(gradArr => tf.stack(gradArr))
          // Concatenate the gradients together, then average them across
          // all the steps of all the games.
          return tf.mean(tf.concat(varGradients, 0), 0)
        })
      }
      return gradients
    });
  }

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
      this.representationModel.save(path + 'rp'),
      this.valueModel.save(path + 'vm'),
      this.policyModel.save(path + 'pm'),
      this.dynamicsModel.save(path + 'dm'),
      this.rewardModel.save(path + 'rm')
    ])
  }

  public async load (path: string): Promise<void> {
    const [
      rp, vm, pm, dm, rm
    ] = await Promise.all([
      tf.loadLayersModel(path + 'rp/model.json'),
      tf.loadLayersModel(path + 'vm/model.json'),
      tf.loadLayersModel(path + 'pm/model.json'),
      tf.loadLayersModel(path + 'dm/model.json'),
      tf.loadLayersModel(path + 'rm/model.json')
    ])
    this.representationModel.setWeights(rp.getWeights())
    this.valueModel.setWeights(vm.getWeights())
    this.policyModel.setWeights(pm.getWeights())
    this.dynamicsModel.setWeights(dm.getWeights())
    this.rewardModel.setWeights(rm.getWeights())
    rp.dispose()
    vm.dispose()
    pm.dispose()
    dm.dispose()
    rm.dispose()
  }

  public copyWeights (network: MuZeroNetwork<Action>): void {
    if (!(network instanceof MuZeroNet)) {
      throw new Error('Incorrect network applied to copy weights')
    }
    tf.tidy(() => {
      network.representationModel.setWeights(this.representationModel.getWeights())
      network.valueModel.setWeights(this.valueModel.getWeights())
      network.policyModel.setWeights(this.policyModel.getWeights())
      network.dynamicsModel.setWeights(this.dynamicsModel.getWeights())
      network.rewardModel.setWeights(this.rewardModel.getWeights())
    })
  }

  public getWeights (): Tensor[] {
    return this.representationModel.getWeights()
        .concat(this.valueModel.getWeights())
        .concat(this.policyModel.getWeights())
        .concat(this.dynamicsModel.getWeights())
        .concat(this.rewardModel.getWeights())
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
  private valueLoss (prediction: tf.Tensor, target: tf.Tensor): tf.Scalar {
    return tf.losses.meanSquaredError(target, prediction, tf.scalar(this.valueScale)).asScalar()
  }

  /**
   * MSE in board games, cross entropy between categorical values in Atari
   * @param prediction
   * @param target
   * @private
   */
  private scalarLoss (prediction: Tensor, target: Tensor): Scalar {
    return tf.losses.meanSquaredError(target, prediction).asScalar()
  }

  /**
   * Scales the gradient for the backward pass
   * @param tensor
   * @param scale
   * @private
   */
  private scaleGradient (tensor: Tensor, scale: number): Tensor {
    // Perform the operation: tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
    return tf.tidy(() => {
      const tidyTensor = tf.variable(tensor, false)
      const scaledGradient = tensor.mul(scale).add(tidyTensor.mul(1 - scale))
      tidyTensor.dispose()
      return scaledGradient
    })
  }
}
