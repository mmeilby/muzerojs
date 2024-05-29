export class Normalizer {
  private range = 0

  constructor (
    private min = Infinity,
    private max = -Infinity
  ) {
    if (this.min < this.max) {
      this.range = this.max - this.min
    }
  }

  update (value: number): void {
    this.min = Math.min(this.min, value)
    this.max = Math.max(this.max, value)
    this.range = this.max - this.min
  }

  normalize (value: number): number {
    return this.range > 0 ? (value - this.min) / this.range : value
  }
}
