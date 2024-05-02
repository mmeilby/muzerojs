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
   * Creates a convolutional neural network (Convnet) for the MNIST data.
   *
   * @returns {tf.LayersModel} An instance of tf.LayersModel.
   */
  private createConvModel (): tf.LayersModel {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential()
    const IMAGE_H = 28
    const IMAGE_W = 28
    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_H, IMAGE_W, 1],
      kernelSize: 3,
      filters: 16,
      activation: 'relu'
    }))

    // After the first layer we include a MaxPooling layer. This acts as a sort of
    // downsampling using max values in a region instead of averaging.
    // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    model.add(tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2
    }))

    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }))

    // Max pooling again.
    model.add(tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2
    }))

    // Add another conv2d layer.
    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }))

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({}))

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }))

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({
      units: 10,
      activation: 'softmax'
    }))

    return model
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
