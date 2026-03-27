export type DisplayMode = 'raw' | 'average' | 'max_hold' | 'min_hold'

export interface SpectrumConfig {
  center_freq: number // MHz
  bandwidth: number // MHz
  fft_size: number
  fps: number
  display_mode: DisplayMode
  averaging_count: number
}

export const DEFAULT_CONFIG: SpectrumConfig = {
  center_freq: 100.0,
  bandwidth: 2.4,
  fft_size: 1024,
  fps: 20,
  display_mode: 'raw',
  averaging_count: 10,
}

export const MESSAGE_TYPE = {
  SPECTRUM: 0x01,
  ALERT: 0x02,
} as const
