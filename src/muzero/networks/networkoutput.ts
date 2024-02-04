import { MuZeroHiddenState } from './nnet'
import {scalar, Tensor, tensor} from "@tensorflow/tfjs-node";

export class NetworkOutput {
  public tfValue: Tensor
  public tfReward: Tensor
  public tfPolicy: Tensor
  public tfHiddenState: Tensor
  constructor (
    public nValue: number,
    public nReward: number,
    public policyMap: number[],
    public aHiddenState: MuZeroHiddenState
  ) {
    this.tfValue = scalar(0)
    this.tfReward = scalar(0)
    this.tfPolicy = tensor(0)
    this.tfHiddenState = tensor(0)
  }
}
