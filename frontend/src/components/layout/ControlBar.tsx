import { Radio, Square, Play } from 'lucide-react'
import type { SpectrumConfig, DisplayMode } from '../../types/signal'

interface Props {
  config: SpectrumConfig
  streaming: boolean
  connected: boolean
  onConfigChange: (config: SpectrumConfig) => void
  onToggleStream: () => void
}

export function ControlBar({
  config,
  streaming,
  connected,
  onConfigChange,
  onToggleStream,
}: Props) {
  return (
    <div className="flex items-center gap-4 bg-gray-900 border-b border-gray-800 px-4 py-2">
      <div className="flex items-center gap-2">
        <Radio className="w-5 h-5 text-cyan-400" />
        <span className="font-semibold text-sm">RF Sub-Band Monitor</span>
      </div>

      <div className="h-4 w-px bg-gray-700" />

      <button
        onClick={onToggleStream}
        className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium ${
          streaming
            ? 'bg-red-900/50 text-red-300 hover:bg-red-900/70'
            : 'bg-emerald-900/50 text-emerald-300 hover:bg-emerald-900/70'
        }`}
      >
        {streaming ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
        {streaming ? 'Stop' : 'Start'}
      </button>

      <div
        className={`w-2 h-2 rounded-full ${
          connected ? 'bg-emerald-400' : 'bg-gray-600'
        }`}
        title={connected ? 'Connected' : 'Disconnected'}
      />

      <div className="h-4 w-px bg-gray-700" />

      <label className="flex items-center gap-1.5 text-xs text-gray-400">
        Center
        <input
          type="number"
          step="0.1"
          value={config.center_freq}
          onChange={e =>
            onConfigChange({ ...config, center_freq: parseFloat(e.target.value) || 100 })
          }
          className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200"
        />
        <span>MHz</span>
      </label>

      <label className="flex items-center gap-1.5 text-xs text-gray-400">
        BW
        <input
          type="number"
          step="0.1"
          value={config.bandwidth}
          onChange={e =>
            onConfigChange({ ...config, bandwidth: parseFloat(e.target.value) || 2.4 })
          }
          className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200"
        />
        <span>MHz</span>
      </label>

      <label className="flex items-center gap-1.5 text-xs text-gray-400">
        FFT
        <select
          value={config.fft_size}
          onChange={e =>
            onConfigChange({ ...config, fft_size: parseInt(e.target.value) })
          }
          className="bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200"
        >
          <option value="512">512</option>
          <option value="1024">1024</option>
          <option value="2048">2048</option>
          <option value="4096">4096</option>
        </select>
      </label>

      <label className="flex items-center gap-1.5 text-xs text-gray-400">
        FPS
        <input
          type="range"
          min="5"
          max="30"
          value={config.fps}
          onChange={e =>
            onConfigChange({ ...config, fps: parseInt(e.target.value) })
          }
          className="w-16"
        />
        <span className="w-4 text-right">{config.fps}</span>
      </label>

      <label className="flex items-center gap-1.5 text-xs text-gray-400">
        Mode
        <select
          value={config.display_mode}
          onChange={e =>
            onConfigChange({ ...config, display_mode: e.target.value as DisplayMode })
          }
          className="bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200"
        >
          <option value="raw">Raw</option>
          <option value="average">Average</option>
          <option value="max_hold">Max Hold</option>
          <option value="min_hold">Min Hold</option>
        </select>
      </label>
    </div>
  )
}
