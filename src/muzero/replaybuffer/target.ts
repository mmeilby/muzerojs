export class MuZeroTarget {
  constructor (
    public readonly value: number,
    public readonly reward: number,
    public readonly policy: number[]
  ) {}
}
