import { useState, useCallback } from 'react'
import './App.css'
import { ControlBar } from './components/layout/ControlBar'
import { SpectrumCanvas } from './components/spectrum/SpectrumCanvas'
import { WaterfallCanvas } from './components/spectrum/WaterfallCanvas'
import { SubBandEditor } from './components/config/SubBandEditor'
import { AlertPanel } from './components/alerts/AlertPanel'
import { useSignalStream } from './hooks/useSignalStream'
import { useSubBands } from './hooks/useSubBands'
import { DEFAULT_CONFIG } from './types/signal'
import type { SpectrumConfig } from './types/signal'
import { api } from './services/api'

function App() {
  const [config, setConfig] = useState<SpectrumConfig>(DEFAULT_CONFIG)
  const [streaming, setStreaming] = useState(false)
  const [pendingRange, setPendingRange] = useState<{ start: number; end: number } | null>(null)

  const { currentSpectrum, waterfallBuffer, connected, alerts, frameCount } =
    useSignalStream(config.fft_size, streaming)
  const { subbands, addSubBand, deleteSubBand } = useSubBands()

  const handleConfigChange = useCallback(
    async (newConfig: SpectrumConfig) => {
      setConfig(newConfig)
      if (streaming) {
        await api.updateConfig(newConfig)
      }
    },
    [streaming]
  )

  const handleToggleStream = useCallback(async () => {
    if (!streaming) {
      await api.updateConfig(config)
    }
    setStreaming(s => !s)
  }, [streaming, config])

  const handleSubBandDrag = useCallback((freqStart: number, freqEnd: number) => {
    setPendingRange({ start: freqStart, end: freqEnd })
  }, [])

  return (
    <div className="min-h-screen flex flex-col bg-gray-950">
      <ControlBar
        config={config}
        streaming={streaming}
        connected={connected}
        onConfigChange={handleConfigChange}
        onToggleStream={handleToggleStream}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Main visualization area */}
        <div className="flex-1 flex flex-col p-3 gap-2 min-w-0">
          <SpectrumCanvas
            spectrum={currentSpectrum}
            config={config}
            subbands={subbands}
            frameCount={frameCount}
            onSubBandDrag={handleSubBandDrag}
          />
          <WaterfallCanvas
            waterfallBuffer={waterfallBuffer}
            frameCount={frameCount}
          />
          {!streaming && !currentSpectrum && (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
              Press Start to begin signal monitoring
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-72 shrink-0 border-l border-gray-800 p-3 space-y-3 overflow-y-auto">
          <SubBandEditor
            subbands={subbands}
            onAdd={addSubBand}
            onDelete={deleteSubBand}
            pendingRange={pendingRange}
            onClearPending={() => setPendingRange(null)}
          />
          <AlertPanel alerts={alerts} />
        </div>
      </div>
    </div>
  )
}

export default App
