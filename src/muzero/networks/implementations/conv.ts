import * as tf from '@tensorflow/tfjs-node'
import { type Model } from '../model'

import debugFactory from 'debug'

const debug = debugFactory('muzero:network:model')

export class ResNet implements Model {
  private readonly representationModel: tf.LayersModel
  private readonly valueModel: tf.LayersModel
  private readonly policyModel: tf.LayersModel
  private readonly dynamicsModel: tf.LayersModel
  private readonly rewardModel: tf.LayersModel

  constructor (
    private readonly inputSize: number[],
    // Length of the action tensors
    private readonly actionSpaceN: number,
    // Length of the hidden state tensors (number of outputs for g.s and h.s)
    protected readonly hxSize: number = 16
  ) {
    this.representationModel = this.makeResNet('hs', this.inputSize, this.hxSize)
    this.representationModel.summary()
    this.valueModel = this.makeValue('fv', [this.hxSize, 1, 1])
    this.valueModel.summary()
    this.policyModel = this.makePolicy('fp', [this.hxSize, 1, 1], this.actionSpaceN)
    this.policyModel.summary()
    this.rewardModel = this.makeValue('dr', [this.hxSize + this.actionSpaceN, 1, 1])
    this.rewardModel.summary()
    this.dynamicsModel = this.makeResNet('ds', [this.hxSize + this.actionSpaceN, 1, 1], this.hxSize)
    this.dynamicsModel.summary()
    debug('Constructed five residual networks (ResNet 18)')
  }

  // Batch normalisation and ReLU always go together, let's add them to the separate function
  private batchNormRelu (name: string, input: tf.SymbolicTensor): tf.SymbolicTensor {
    const batch = tf.layers.batchNormalization({
      name: `${name}_bn`
    }).apply(input)
    return tf.layers.reLU({
      name: `${name}_rl`
    }).apply(batch) as tf.SymbolicTensor
  }

  /**
   *
   * @param name
   * @param input
   * @param filters
   * @param noDownSample
   * @private
   */
  private makeResidualBlock (name: string, input: tf.SymbolicTensor, filters: number, noDownSample: boolean = false): tf.SymbolicTensor {
    const filter1 = tf.layers.conv2d({
      name: `${name}_f1_cv`,
      kernelSize: 3,
      filters,
      padding: 'same',
      strides: noDownSample ? 1 : 2,
      kernelInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    const filter1norm = this.batchNormRelu(`${name}_f1`, filter1)
    // Residual connection - here we sum up first matrix and the result of 2 convolutions
    return tf.layers.add({
      name: `${name}_ad`
    }).apply([filter1, filter1norm]) as tf.SymbolicTensor
  }

  // ResNet - put all together
  private makeResNet (name: string, inputShape: number[], outputSize: number): tf.LayersModel {
    const input = tf.input({
      shape: inputShape,
      name: `${name}_in`
    })
    const conv1Filter = tf.layers.conv2d({
      name: `${name}_f1_cv`,
      kernelSize: 3,
      filters: inputShape[2],
      strides: 1,
      padding: 'same',
      kernelInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    let filter1norm = this.batchNormRelu(`${name}_f1`, conv1Filter)
    for (let i = 1; i <= 8; i++) {
      filter1norm = this.makeResidualBlock(`${name}_rb${i}`, filter1norm, inputShape[2] * Math.pow(2, i), true)
    }
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    }).apply(filter1norm)
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: outputSize,
      kernelInitializer: 'glorotNormal',
      activation: 'relu'
    }).apply(flatten) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: dense
    })
  }

  private makePolicy (name: string, inputShape: number[], outputSize: number): tf.LayersModel {
    const input = tf.input({
      shape: inputShape,
      name: `${name}_in`
    })
    const conv1Filter = tf.layers.conv2d({
      name: `${name}_f1_cv`,
      kernelSize: 1,
      filters: inputShape[2] * 2,
      strides: 1,
      padding: 'same',
      kernelInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    const filter1norm = this.batchNormRelu(`${name}_f1`, conv1Filter)
    const conv2Filter = tf.layers.conv2d({
      name: `${name}_f2_cv`,
      kernelSize: 1,
      filters: 1,
      strides: 1,
      padding: 'same',
      kernelInitializer: 'glorotNormal'
    }).apply(filter1norm) as tf.SymbolicTensor
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    }).apply(conv2Filter)
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: outputSize,
      kernelInitializer: 'glorotNormal',
      activation: 'softmax'
    }).apply(flatten) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: dense
    })
  }

  private makeValue (name: string, inputShape: number[]): tf.LayersModel {
    const input = tf.input({
      shape: inputShape,
      name: `${name}_in`
    })
    const conv1Filter = tf.layers.conv2d({
      name: `${name}_f1_cv`,
      kernelSize: 1,
      filters: inputShape[2] * 2,
      strides: 1,
      padding: 'same',
      kernelInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    const filter1norm = this.batchNormRelu(`${name}_f1`, conv1Filter)
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    }).apply(filter1norm)
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: 1,
      kernelInitializer: 'glorotNormal',
      activation: 'tanh'
    }).apply(flatten) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: dense
    })
  }

  public representation (observation: tf.Tensor): tf.Tensor {
    return this.representationModel.predict(observation) as tf.Tensor
  }

  public value (state: tf.Tensor): tf.Tensor {
    return this.valueModel.predict(state.expandDims(2).expandDims(3)) as tf.Tensor
  }

  public policy (state: tf.Tensor): tf.Tensor {
    return this.policyModel.predict(state.expandDims(2).expandDims(3)) as tf.Tensor
  }

  public dynamics (conditionedState: tf.Tensor): tf.Tensor {
    return this.dynamicsModel.predict(conditionedState.expandDims(2).expandDims(3)) as tf.Tensor
  }

  public reward (conditionedState: tf.Tensor): tf.Tensor {
    return this.rewardModel.predict(conditionedState.expandDims(2).expandDims(3)) as tf.Tensor
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

  public copyWeights (network: Model): void {
    if (network instanceof ResNet) {
      tf.tidy(() => {
        network.representationModel.setWeights(this.representationModel.getWeights())
        network.valueModel.setWeights(this.valueModel.getWeights())
        network.policyModel.setWeights(this.policyModel.getWeights())
        network.dynamicsModel.setWeights(this.dynamicsModel.getWeights())
        network.rewardModel.setWeights(this.rewardModel.getWeights())
      })
    } else {
      throw new Error(`ResNet: Cant copy weights to a different model: ${network.constructor.name}`)
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
