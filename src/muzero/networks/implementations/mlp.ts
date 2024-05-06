import * as tf from '@tensorflow/tfjs-node-gpu'
// import debugFactory from 'debug'
import { type Model } from '../model'

// const debug = debugFactory('muzero:network:debug')

export class MlpNet implements Model {
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

  private readonly logDir: string

  private representationModel: tf.Sequential = tf.sequential()
  private valueModel: tf.Sequential = tf.sequential()
  private policyModel: tf.Sequential = tf.sequential()
  private dynamicsModel: tf.Sequential = tf.sequential()
  private rewardModel: tf.Sequential = tf.sequential()

  constructor (
    private readonly inputSize: number[],
    // Length of the action tensors
    private readonly actionSpaceN: number,
    hiddenLayerSizes: number | number[] = [16, 16]
  ) {
    // hidden state size
    this.hxSize = 16
    this.rewardSupportSize = 0
    this.valueSupportSize = 0
    this.hiddenLayerSize = !Array.isArray(hiddenLayerSizes) ? [hiddenLayerSizes] : hiddenLayerSizes
    this.valueScale = 0.25
    this.weightDecay = 0.000005

    this.logDir = './logs/20230109-005200' // + sysdatetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    this.makeModel()
  }

  public makeModel (): void {
    const repModel = tf.sequential()
    const inputShape = [this.inputSize.reduce((m, v) => m * v, 1)]
    this.makeHiddenLayer(repModel, 'representation_state_hidden', inputShape)
    repModel.add(tf.layers.dense({
      name: 'representation_state_output',
      units: this.hxSize,
      activation: 'tanh',
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
      activation: 'linear',
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
      activation: 'linear',
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
      activation: 'tanh',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: this.weightDecay }),
      useBias: false
    }))
    this.dynamicsModel = dynamicsModel
  }

  private makeHiddenLayer (model: tf.Sequential, name: string, inputShape: number[]): void {
    this.hiddenLayerSize.forEach((units, i) => {
      model.add(tf.layers.dense({
        name: `${name}${i + 1}`,
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
   * Predict the MuZero hidden state corresponding to the game state (observation).
   * The observation may be multidimensional, so we need to flatten it for the MLP first layer.
   * However, we should preserve the batch dimension to allow more observations to be processed in one prediction
   * @param observation A tensor representing the game state
   * @returns Corresponding MuZero hidden state for each observation
   */
  public representation (observation: tf.Tensor): tf.Tensor {
    // Get the number of observations to be processed and flatten preserving the batch size
    const batchSize = observation.shape[0]
    return this.representationModel.predict(observation.reshape([batchSize, -1])) as tf.Tensor
  }

  public value (state: tf.Tensor): tf.Tensor {
    return this.valueModel.predict(state) as tf.Tensor
  }

  public policy (state: tf.Tensor): tf.Tensor {
    return this.policyModel.predict(state) as tf.Tensor
  }

  public dynamics (conditionedState: tf.Tensor): tf.Tensor {
    return this.dynamicsModel.predict(conditionedState) as tf.Tensor
  }

  public reward (conditionedState: tf.Tensor): tf.Tensor {
    return this.rewardModel.predict(conditionedState) as tf.Tensor
  }

  /**
   * initialInference
   * Execute h(o)->s f(s)->p,v
   * @param obs
   *
   private initialInference (obs: Observation): NetworkOutput {
   if (!(obs instanceof NetworkObservation)) {
   throw new Error('Incorrect observation applied to initialInference')
   }
   const observation = tf.tensor1d(obs.observation).reshape([1, -1])
   const tfHiddenState = this.representationModel.predict(observation) as tf.Tensor
   const hiddenState: NetworkHiddenState = new NetworkHiddenState(tfHiddenState.reshape([-1]).arraySync() as number[])
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
   *
   private recurrentInference (hiddenState: HiddenState, action: Actionwise): NetworkOutput {
   if (!(hiddenState instanceof NetworkHiddenState)) {
   throw new Error('Incorrect hidden state applied to recurrentInference')
   }
   const conditionedHiddenState = tf.concat([tf.tensor2d([hiddenState.state]), this.policyTransform(action.id)], 1)
   const tfHiddenState = this.dynamicsModel.predict(conditionedHiddenState) as tf.Tensor
   const newHiddenState: NetworkHiddenState = new NetworkHiddenState(tfHiddenState.reshape([-1]).arraySync() as number[])
   const tfReward = this.rewardModel.predict(conditionedHiddenState) as tf.Tensor
   const reward = this.inverseRewardTransform(tfReward)
   const tfPolicy = this.policyModel.predict(tfHiddenState) as tf.Tensor
   const policy = this.inversePolicyTransform(tfPolicy)
   const tfValue = this.valueModel.predict(tfHiddenState) as tf.Tensor
   const value = this.inverseValueTransform(tfValue)
   return new NetworkOutput(value, reward, policy, newHiddenState)
   }
   *
   /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   *
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
   *
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
   *
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
   })
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
   *
   private valueTransform (value: number): tf.Tensor {
   return scalarToSupport([value], this.valueSupportSize)
   }

   /*
   private trainingSteps (): number {
   return 1
   }
   *
   private valueLoss (prediction: tf.Tensor, target: tf.Tensor): tf.Scalar {
   return tf.losses.meanSquaredError(target, prediction, tf.scalar(this.valueScale)).asScalar()
   }

   /**
   * MSE in board games, cross entropy between categorical values in Atari
   * @param prediction
   * @param target
   * @private
   *
   private scalarLoss (prediction: Tensor, target: Tensor): Scalar {
   return tf.losses.meanSquaredError(target, prediction).asScalar()
   }

   /**
   * Scales the gradient for the backward pass
   * @param tensor
   * @param scale
   * @private
   *
   private scaleGradient (tensor: Tensor, scale: number): Tensor {
   // Perform the operation: tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
   return tf.tidy(() => {
   const tidyTensor = tf.variable(tensor, false)
   const scaledGradient = tensor.mul(scale).add(tidyTensor.mul(1 - scale))
   tidyTensor.dispose()
   return scaledGradient
   })
   }
   */
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

  public copyWeights (network: Model): void {
    if (network instanceof MlpNet) {
      tf.tidy(() => {
        network.representationModel.setWeights(this.representationModel.getWeights())
        network.valueModel.setWeights(this.valueModel.getWeights())
        network.policyModel.setWeights(this.policyModel.getWeights())
        network.dynamicsModel.setWeights(this.dynamicsModel.getWeights())
        network.rewardModel.setWeights(this.rewardModel.getWeights())
      })
    } else {
      throw new Error(`ConvNet: Cant copy weights from a different model: ${Object.getOwnPropertyNames(network).join(',')}`)
    }
  }

  public dispose (): number {
    let disposed = 0
    disposed += this.representationModel.dispose().numDisposedVariables
    disposed += this.valueModel.dispose().numDisposedVariables
    disposed += this.policyModel.dispose().numDisposedVariables
    disposed += this.dynamicsModel.dispose().numDisposedVariables
    disposed += this.rewardModel.dispose().numDisposedVariables
    return disposed
  }
}
