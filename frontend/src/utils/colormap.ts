/**
 * Viridis colormap lookup table.
 * Maps normalized value (0-1) to [R, G, B] (0-255).
 */
const VIRIDIS_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [32, 144, 140],
  [34, 167, 132],
  [68, 190, 112],
  [121, 209, 81],
  [189, 222, 38],
  [253, 231, 37],
]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

export function viridis(value: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, value))
  const scaled = v * (VIRIDIS_STOPS.length - 1)
  const idx = Math.floor(scaled)
  const t = scaled - idx

  if (idx >= VIRIDIS_STOPS.length - 1) {
    return VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1]
  }

  const a = VIRIDIS_STOPS[idx]
  const b = VIRIDIS_STOPS[idx + 1]

  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
  ]
}

// Pre-computed lookup table for fast rendering (256 entries)
export const VIRIDIS_LUT: Uint8Array = (() => {
  const lut = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = viridis(i / 255)
    lut[i * 3] = r
    lut[i * 3 + 1] = g
    lut[i * 3 + 2] = b
  }
  return lut
})()
