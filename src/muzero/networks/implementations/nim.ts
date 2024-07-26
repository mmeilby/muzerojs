import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Model } from '../model'

import debugFactory from 'debug'
import type { Config } from '../../games/core/config'

export class NimNet implements Model {
  // Shape of the hidden state tensors (outputs for g.s and h.s)
  protected readonly hxShape: number[]
  private readonly representationModel: tf.LayersModel
  private readonly valueModel: tf.LayersModel
  private readonly policyModel: tf.LayersModel
  private readonly dynamicsModel: tf.LayersModel
  private readonly rewardModel: tf.LayersModel
  private readonly inputShape: number[]

  constructor (
    private readonly config: Config
  ) {
    this.inputShape = this.config.observationSize
    this.hxShape = this.config.observationSize
    if (this.config.supervisedRL) {
      this.valueModel = this.makeSupervisedValue('fv', this.inputShape)
      this.policyModel = this.makeSupervisedPolicy('fp', this.inputShape, this.config.actionSpace)
      this.representationModel = tf.sequential()
      this.dynamicsModel = tf.sequential()
      this.rewardModel = tf.sequential()
    } else {
      const condHxShape: number[] = [...this.hxShape]
      condHxShape[this.hxShape.length - 1]++
      this.representationModel = this.makeState('hs', this.inputShape, this.hxShape)
      this.valueModel = this.makeValue('fv', this.hxShape)
      this.policyModel = this.makePolicy('fp', this.hxShape, this.config.actionSpace)
      this.rewardModel = this.makeValue('dr', condHxShape)
      this.dynamicsModel = this.makeState('ds', condHxShape, this.hxShape)
    }
  }

  public representation (observation: tf.Tensor): tf.Tensor {
    return this.representationModel.predict(observation) as tf.Tensor
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

  public async trainPolicy (x: tf.Tensor, y: tf.Tensor): Promise<tf.History> {
    //    debug(`trainPolicy: ${x.toString()} ${y.toString()}`)
    const param: tf.ModelFitArgs = {
      batchSize: this.config.batchSize,
      epochs: this.config.epochs,
      verbose: 0
      //      validationSplit: 0.15
    }
    return await this.policyModel.fit(x, y, param)
  }

  public async trainValue (x: tf.Tensor, y: tf.Tensor): Promise<tf.History> {
    //    debug(`trainValue: ${x.toString()} ${y.toString()}`)
    const param: tf.ModelFitArgs = {
      batchSize: this.config.batchSize,
      epochs: this.config.epochs,
      verbose: 0
      //      validationSplit: 0.15 // 15% batch data will be used for validation
    }
    return await this.valueModel.fit(x, y, param)
  }

  public async save (path: string): Promise<void> {
    if (this.config.supervisedRL) {
      await Promise.all([
        this.valueModel.save(path + 'vm'),
        this.policyModel.save(path + 'pm')
      ])
    } else {
      await Promise.all([
        this.representationModel.save(path + 'rp'),
        this.valueModel.save(path + 'vm'),
        this.policyModel.save(path + 'pm'),
        this.dynamicsModel.save(path + 'dm'),
        this.rewardModel.save(path + 'rm')
      ])
    }
  }

  public async load (path: string): Promise<void> {
    if (this.config.supervisedRL) {
      const [
        vm, pm
      ] = await Promise.all([
        tf.loadLayersModel(path + 'vm/model.json'),
        tf.loadLayersModel(path + 'pm/model.json')
      ])
      this.valueModel.setWeights(vm.getWeights())
      this.policyModel.setWeights(pm.getWeights())
      vm.dispose()
      pm.dispose()
    } else {
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
  }

  public copyWeights (network: Model): void {
    if (network instanceof NimNet) {
      tf.tidy(() => {
        if (!this.config.supervisedRL) {
          network.representationModel.setWeights(this.representationModel.getWeights())
          network.dynamicsModel.setWeights(this.dynamicsModel.getWeights())
          network.rewardModel.setWeights(this.rewardModel.getWeights())
        }
        network.valueModel.setWeights(this.valueModel.getWeights())
        network.policyModel.setWeights(this.policyModel.getWeights())
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

  public print (): void {
    const output = debugFactory('muzero:muzero:output')
    if (!this.config.supervisedRL) {
      this.representationModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
        output(msg)
      })
    }
    this.policyModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      output(msg)
    })
    this.valueModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      output(msg)
    })
    if (!this.config.supervisedRL) {
      this.dynamicsModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
        output(msg)
      })
      this.rewardModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
        output(msg)
      })
    }
  }

  /*
  class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

  class ResidualBlock(nn.Module):
      def __init__(self, filters):
          super().__init__()
          self.conv = Conv(filters, filters, 3, True)

      def forward(self, x):
          return F.relu(x + (self.conv(x)))

   */
  private makeConv (name: string, input: tf.SymbolicTensor, filters: number, kernelSize: number, bn: boolean = false): tf.SymbolicTensor {
    const conv = tf.layers.conv2d({
      name: `${name}_2d`,
      kernelSize,
      filters,
      padding: 'same'
    })
    const batchNorm = tf.layers.batchNormalization({
      name: `${name}_bn`
    })
    const layer = bn ? batchNorm.apply(conv.apply(input)) : conv.apply(input)
    return layer as tf.SymbolicTensor
  }

  /*
  num_filters = 16
  num_blocks = 4

  class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(x).unsqueeze(0))
        return rp.cpu().numpy()[0]

  class Prediction(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, action_shape):
        super().__init__()
        self.board_size = np.prod(action_shape[1:])
        self.action_size = action_shape[0] * self.board_size

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, rp):
        h_p = F.relu(self.conv_p1(rp))
        h_p = self.conv_p2(h_p).view(-1, self.action_size)

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # range of value is -1 ~ 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def inference(self, rp):
        self.eval()
        with torch.no_grad():
            p, v = self(torch.from_numpy(rp).unsqueeze(0))
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]

   */

  private makeState (name: string, inputShape: number[], outputShape: number[]): tf.LayersModel {
    const input = tf.layers.input({
      name: `${name}_in`,
      shape: inputShape
    })
    //    const conv = this.makeConv(`${name}_rb0`, input, 16, 3, true)
    const conv = this.makeConv(`${name}_rb`, input, outputShape[outputShape.length - 1], 3, true)
    const relu = tf.layers.reLU({
      name: `${name}_rl0`
    }).apply(conv) as tf.SymbolicTensor
    /*
        const rsb1 = this.makeResidualBlock(`${name}_rb1`, relu, 16)
        const rsb2 = this.makeResidualBlock(`${name}_rb2`, rsb1, 16)
        const rsb3 = this.makeResidualBlock(`${name}_rb3`, rsb2, 16)
        const rsb4 = this.makeResidualBlock(`${name}_rb4`, rsb3, 16)
        const conv2 = this.makeConv(`${name}_rb5`, rsb4, outputShape[outputShape.length - 1], 3)
        const relu2 = tf.layers.reLU({
          name: `${name}_rl5`
        }).apply(conv2) as tf.SymbolicTensor
    */
    return tf.model({
      inputs: input,
      outputs: relu
    })
  }

  private makePolicy (name: string, inputShape: number[], outputSize: number): tf.LayersModel {
    const input = tf.layers.input({
      name: `${name}_in`,
      shape: inputShape
    })
    const conv = this.makeConv(`${name}_fi1`, input, 4, 1, true)
    const relu = tf.layers.reLU({
      name: `${name}_rl`
    }).apply(conv) as tf.SymbolicTensor
    const conv2 = this.makeConv(`${name}_fi2`, relu, 1, 1)
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    })
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: outputSize,
      activation: 'softmax'
    }).apply(flatten.apply(conv2)) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: dense
    })
  }

  private makeValue (name: string, inputShape: number[]): tf.LayersModel {
    const input = tf.layers.input({
      name: `${name}_in`,
      shape: inputShape
    })
    const conv = this.makeConv(`${name}`, input, 4, 1, true)
    const relu = tf.layers.reLU({
      name: `${name}_rl`
    }).apply(conv) as tf.SymbolicTensor
    const flatten = tf.layers.flatten({
      name: `${name}_fl`
    })
    const dense = tf.layers.dense({
      name: `${name}_de`,
      units: 1,
      activation: 'tanh'
    }).apply(flatten.apply(relu)) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: dense
    })
  }

  private makeSupervisedPolicy (name: string, inputShape: number[], outputSize: number): tf.LayersModel {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential()

    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
      name: `${name}_c2d1`,
      inputShape,
      kernelSize: 3,
      filters: 16,
      padding: 'same',
      activation: 'relu'
    }))

    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({
      name: `${name}_c2d2`,
      kernelSize: 3,
      filters: 32,
      padding: 'same',
      activation: 'relu'
    }))

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({
      name: `${name}_fl`
    }))

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({
      name: `${name}_de`,
      units: outputSize,
      activation: 'softmax'
    }))
    model.compile({
      loss: tf.losses.softmaxCrossEntropy,
      metrics: ['accuracy'],
      optimizer: tf.train.rmsprop(this.config.lrInit, this.config.lrDecayRate, this.config.momentum)
    })

    return model
  }

  private makeSupervisedValue (name: string, inputShape: number[]): tf.LayersModel {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential()

    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
      name: `${name}_c2d1`,
      inputShape,
      kernelSize: 3,
      filters: 16,
      padding: 'same',
      activation: 'relu'
    }))

    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({
      name: `${name}_c2d2`,
      kernelSize: 3,
      filters: 32,
      padding: 'same',
      activation: 'relu'
    }))

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({
      name: `${name}_fl`
    }))

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({
      name: `${name}_de`,
      units: 1,
      activation: 'tanh'
    }))
    model.compile({
      loss: tf.losses.meanSquaredError,
      metrics: ['accuracy'],
      optimizer: tf.train.rmsprop(this.config.lrInit, this.config.lrDecayRate, this.config.momentum)
    })

    return model
  }
}
