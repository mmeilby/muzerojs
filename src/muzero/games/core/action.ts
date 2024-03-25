import { type Actionwise } from '../../selfplay/entities'

export class MuZeroAction implements Actionwise {
  private readonly index: number

  public constructor (id: number) {
    this.index = id
  }

  get id (): number {
    return this.index
  }
}
