import { MuZeroHiddenState } from './nnet'

export class NetworkOutput {
  constructor (
    public nValue: number,
    public nReward: number,
    public policyMap: number[],
    public aHiddenState: MuZeroHiddenState
  ) {}
}
