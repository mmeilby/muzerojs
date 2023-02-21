import { Actionwise } from '../core/actionwise'

export class NimAction implements Actionwise {
  public constructor (
    private readonly index: number
  ) {}

  get id (): number {
    return this.index
  }
}
