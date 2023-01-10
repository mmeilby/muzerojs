export class MuZeroPlayer {
  private readonly index: number

  constructor (id: number) {
    this.index = id
  }

  public id (): number {
    return this.index
  }
}
