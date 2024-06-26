import * as tf from '@tensorflow/tfjs-node-gpu'
import { type Model } from '../model'

import debugFactory from 'debug'

const debug = debugFactory('muzero:network:model')

export class NimNet implements Model {
  private readonly representationModel: tf.LayersModel
  private readonly valueModel: tf.LayersModel
  private readonly policyModel: tf.LayersModel
  private readonly dynamicsModel: tf.LayersModel
  private readonly rewardModel: tf.LayersModel

  constructor (
    private readonly inputShape: number[],
    // Length of the policy tensors
    private readonly actionSpaceN: number,
    // Shape of the hidden state tensors (outputs for g.s and h.s)
    protected readonly hxShape: number[],
    // Shape of action tensors
    protected readonly actionShape: number[]
  ) {
    const condHxShape: number[] = [...this.hxShape]
    condHxShape[this.hxShape.length - 1]++
    this.representationModel = this.makeState('hs', this.inputShape, this.hxShape)
    this.representationModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      debug(msg)
    })
    this.valueModel = this.makeValue('fv', this.hxShape)
    this.valueModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      debug(msg)
    })
    this.policyModel = this.makePolicy('fp', this.hxShape, this.actionSpaceN)
    this.policyModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      debug(msg)
    })
    this.rewardModel = this.makeValue('dr', condHxShape)
    this.rewardModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      debug(msg)
    })
    this.dynamicsModel = this.makeState('ds', condHxShape, this.hxShape)
    this.dynamicsModel.summary(100, [25, 50, 70, 80, 100], (msg) => {
      debug(msg)
    })
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

  public async trainRepresentation (labels: tf.Tensor, targets: tf.Tensor): Promise<number | number[]> {
    return await this.representationModel.trainOnBatch(labels, targets)
  }

  public async trainPolicy (labels: tf.Tensor, targets: tf.Tensor): Promise<number | number[]> {
    return await this.policyModel.trainOnBatch(labels, targets)
  }

  public async trainValue (labels: tf.Tensor, targets: tf.Tensor): Promise<number | number[]> {
    return await this.valueModel.trainOnBatch(labels, targets)
  }

  public async trainDynamics (labels: tf.Tensor, targets: tf.Tensor): Promise<number | number[]> {
    return await this.dynamicsModel.trainOnBatch(labels, targets)
  }

  public async trainReward (labels: tf.Tensor, targets: tf.Tensor): Promise<number | number[]> {
    return await this.rewardModel.trainOnBatch(labels, targets)
  }

  public getHiddenStateWeights (): tf.Variable[] {
    const trainableWeights = this.representationModel.trainableWeights.concat(this.dynamicsModel.trainableWeights)
    return trainableWeights.map(w => new tf.Variable(w.read(), true, w.name, w.id))
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
    if (network instanceof NimNet) {
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

  /**
   *
   * @param name
   * @param input
   * @param filters
   * @param noDownSample
   * @private
   */
  private makeResidualBlock (name: string, input: tf.SymbolicTensor, filters: number): tf.SymbolicTensor {
    const conv = this.makeConv(`${name}`, input, filters, 3, true)
    // Residual connection - here we sum up first matrix and the result of 2 convolutions
    const add = tf.layers.add({
      name: `${name}_ad`
    })
    const relu = tf.layers.reLU({
      name: `${name}_rl`
    })
    return relu.apply(add.apply([input, conv])) as tf.SymbolicTensor
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
    const conv = this.makeConv(`${name}_rb0`, input, 16, 3, true)
    const relu = tf.layers.reLU({
      name: `${name}_rl0`
    }).apply(conv) as tf.SymbolicTensor
    const rsb1 = this.makeResidualBlock(`${name}_rb1`, relu, 16)
    const rsb2 = this.makeResidualBlock(`${name}_rb2`, rsb1, 16)
    const rsb3 = this.makeResidualBlock(`${name}_rb3`, rsb2, 16)
    const rsb4 = this.makeResidualBlock(`${name}_rb4`, rsb3, 16)
    const conv2 = this.makeConv(`${name}_rb5`, rsb4, outputShape[outputShape.length - 1], 3)
    const relu2 = tf.layers.reLU({
      name: `${name}_rl5`
    }).apply(conv2) as tf.SymbolicTensor
    return tf.model({
      inputs: input,
      outputs: relu2
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
}
