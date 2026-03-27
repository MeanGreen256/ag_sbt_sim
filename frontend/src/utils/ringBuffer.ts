/**
 * Ring buffer for waterfall display.
 * Stores the last `rows` spectrum frames for rendering.
 */
export class RingBuffer {
  readonly width: number
  readonly rows: number
  readonly data: Float32Array
  private writeIndex = 0
  count = 0

  constructor(width: number, rows: number) {
    this.width = width
    this.rows = rows
    this.data = new Float32Array(width * rows)
    this.data.fill(-120) // initialize to noise floor
  }

  push(row: Float32Array): void {
    const offset = this.writeIndex * this.width
    this.data.set(row.subarray(0, this.width), offset)
    this.writeIndex = (this.writeIndex + 1) % this.rows
    if (this.count < this.rows) this.count++
  }

  /**
   * Get row by age (0 = most recent, 1 = one before, etc.)
   */
  getRow(age: number): Float32Array {
    if (age >= this.count) {
      return new Float32Array(this.width).fill(-120)
    }
    const idx =
      ((this.writeIndex - 1 - age + this.rows) % this.rows) * this.width
    return this.data.subarray(idx, idx + this.width)
  }
}
