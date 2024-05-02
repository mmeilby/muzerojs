import * as tf from '@tensorflow/tfjs-node'
import { type Model } from '../model'

import debugFactory from 'debug'

const debug = debugFactory('muzero:network:model')

export class ConvNet implements Model {
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
    this.valueModel = this.makeResNet('fv', [this.hxSize, 1, 1], 1)
    this.valueModel.summary()
    this.policyModel = this.makeResNet('fp', [this.hxSize, 1, 1], this.actionSpaceN, true)
    this.policyModel.summary()
    this.rewardModel = this.makeResNet('dr', [this.hxSize + this.actionSpaceN, 1, 1], 1)
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
    const filter1 = tf.layers.separableConv2d({
      name: `${name}_f1_cv`,
      kernelSize: 3,
      filters,
      activation: 'relu',
      padding: 'same',
      strides: noDownSample ? 1 : 2,
      depthwiseInitializer: 'glorotNormal',
      pointwiseInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    const filter1norm = this.batchNormRelu(`${name}_f1`, filter1)
    const filter2 = tf.layers.separableConv2d({
      name: `${name}_f2_cv`,
      kernelSize: 3,
      filters,
      activation: 'relu',
      padding: 'same',
      depthwiseInitializer: 'glorotNormal',
      pointwiseInitializer: 'glorotNormal'
    }).apply(filter1norm) as tf.SymbolicTensor
    const dropout = tf.layers.dropout({
      name: `${name}_f2_do`,
      rate: 0.3
    }).apply(filter2) as tf.SymbolicTensor
    const batchNorm = this.batchNormRelu(`${name}_f2`, dropout)
    // Residual connection - here we sum up first matrix and the result of 2 convolutions
    return tf.layers.add({
      name: `${name}_ad`
    }).apply([filter1, batchNorm]) as tf.SymbolicTensor
  }

  // ResNet - put all together
  private makeResNet (name: string, inputShape: number[], outputSize: number, softmax: boolean = false): tf.LayersModel {
    const shape = [...inputShape]
    // Ensure three dimension inputs
    while (shape.length < 3) {
      shape.push(1)
    }
    const input = tf.input({
      shape,
      name: `${name}_in`
    })
    const conv1Filter = tf.layers.conv2d({
      name: `${name}_f1_cv`,
      kernelSize: 5,
      filters: 16,
      strides: 2,
      activation: 'relu',
      padding: 'same',
      kernelInitializer: 'glorotNormal'
    }).apply(input) as tf.SymbolicTensor
    const conv1 = tf.layers.maxPooling2d({
      name: `${name}_f1_mp`,
      poolSize: [3, 3],
      strides: [2, 2],
      padding: 'same'
    }).apply(this.batchNormRelu(`${name}_f1`, conv1Filter)) as tf.SymbolicTensor

    // conv2
    const residual1 = this.makeResidualBlock(`${name}_rb1`, conv1, 16, true)
    // conv3
    const residual2 = this.makeResidualBlock(`${name}_rb2`, residual1, 32)
    // conv4
    const residual3 = this.makeResidualBlock(`${name}_rb3`, residual2, 64)
    // conv5
    const residual4 = this.makeResidualBlock(`${name}_rb4`, residual3, 128)
    /*
        const conv5 = tf.layers.avgPool2d({
          name: `${name}_f5_ap`,
          poolSize: [8, 8],
          strides: [1, 1]
        }).apply(residual4)
    */
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    }).apply(residual4)
    const dropout = tf.layers.dropout({
      name: `${name}_do`,
      rate: 0.5
    }).apply(flatten)
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: outputSize,
      kernelInitializer: 'glorotNormal',
      activation: softmax ? 'softmax' : 'tanh'
    }).apply(dropout) as tf.SymbolicTensor
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
    if (network instanceof ConvNet) {
      tf.tidy(() => {
        network.representationModel.setWeights(this.representationModel.getWeights())
        network.valueModel.setWeights(this.valueModel.getWeights())
        network.policyModel.setWeights(this.policyModel.getWeights())
        network.dynamicsModel.setWeights(this.dynamicsModel.getWeights())
        network.rewardModel.setWeights(this.rewardModel.getWeights())
      })
    } else {
      throw new Error(`ConvNet: Cant copy weights to a different model: ${network.constructor.name}`)
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
