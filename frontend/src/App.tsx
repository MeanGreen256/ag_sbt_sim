import { useState, useCallback } from 'react'
import './App.css'
import { ControlBar } from './components/layout/ControlBar'
import { SpectrumCanvas } from './components/spectrum/SpectrumCanvas'
import { WaterfallCanvas } from './components/spectrum/WaterfallCanvas'
import { SubBandEditor } from './components/config/SubBandEditor'
import { AlertPanel } from './components/alerts/AlertPanel'
import { EmitterLibrary } from './components/emitters/EmitterLibrary'
import { EmitterTimeline } from './components/emitters/EmitterTimeline'
import { EnrollModal } from './components/config/EnrollModal'
import { useSignalStream } from './hooks/useSignalStream'
import { useSubBands } from './hooks/useSubBands'
import { useEmitters } from './hooks/useEmitters'
import { DEFAULT_CONFIG } from './types/signal'
import type { SpectrumConfig } from './types/signal'
import { api } from './services/api'

function App() {
  const [config, setConfig] = useState<SpectrumConfig>(DEFAULT_CONFIG)
  const [streaming, setStreaming] = useState(false)
  const [pendingRange, setPendingRange] = useState<{ start: number; end: number } | null>(null)
  const [enrollRange, setEnrollRange] = useState<{ start: number; end: number } | null>(null)
  const [showDragChoice, setShowDragChoice] = useState<{ start: number; end: number } | null>(null)

  const { currentSpectrum, waterfallBuffer, connected, alerts, frameCount, matchData } =
    useSignalStream(config.fft_size, streaming)
  const { subbands, addSubBand, deleteSubBand } = useSubBands()
  const { emitters, enrolling, enroll, deleteEmitter } = useEmitters()

  const handleConfigChange = useCallback(
    async (newConfig: SpectrumConfig) => {
      setConfig(newConfig)
      if (streaming) {
        try {
          await api.updateConfig(newConfig)
        } catch {
          // Backend unreachable
        }
      }
    },
    [streaming]
  )

  const handleToggleStream = useCallback(async () => {
    if (!streaming) {
      try {
        await api.updateConfig(config)
      } catch {
        // Backend may not be ready
      }
    }
    setStreaming(s => !s)
  }, [streaming, config])

  const handleSubBandDrag = useCallback((freqStart: number, freqEnd: number) => {
    setShowDragChoice({ start: freqStart, end: freqEnd })
  }, [])

  const handleDragChoiceSubBand = useCallback(() => {
    if (showDragChoice) {
      setPendingRange(showDragChoice)
      setShowDragChoice(null)
    }
  }, [showDragChoice])

  const handleDragChoiceEnroll = useCallback(() => {
    if (showDragChoice) {
      setEnrollRange(showDragChoice)
      setShowDragChoice(null)
    }
  }, [showDragChoice])

  const handleEnroll = useCallback(
    async (name: string, tags: string[]) => {
      if (!enrollRange) return
      await enroll({
        freq_start: enrollRange.start,
        freq_end: enrollRange.end,
        name,
        tags,
      })
      setEnrollRange(null)
    },
    [enrollRange, enroll]
  )

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
            matchData={matchData}
            onSubBandDrag={handleSubBandDrag}
          />
          <WaterfallCanvas
            waterfallBuffer={waterfallBuffer}
            frameCount={frameCount}
          />
          <EmitterTimeline emitters={emitters} matchData={matchData} />
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
          <EmitterLibrary
            emitters={emitters}
            matchData={matchData}
            onDelete={deleteEmitter}
          />
          <AlertPanel alerts={alerts} />
        </div>
      </div>

      {/* Drag choice modal */}
      {showDragChoice && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-64">
            <h3 className="text-sm font-semibold text-gray-200 mb-1">
              {showDragChoice.start.toFixed(3)} – {showDragChoice.end.toFixed(3)} MHz
            </h3>
            <p className="text-xs text-gray-400 mb-3">What would you like to do?</p>
            <div className="flex gap-2">
              <button
                onClick={handleDragChoiceSubBand}
                className="flex-1 px-3 py-1.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-500"
              >
                Create Sub-Band
              </button>
              <button
                onClick={handleDragChoiceEnroll}
                className="flex-1 px-3 py-1.5 text-xs rounded bg-cyan-600 text-white hover:bg-cyan-500"
              >
                Enroll Emitter
              </button>
            </div>
            <button
              onClick={() => setShowDragChoice(null)}
              className="w-full mt-2 px-3 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Enrollment modal */}
      {enrollRange && (
        <EnrollModal
          freqStart={enrollRange.start}
          freqEnd={enrollRange.end}
          enrolling={enrolling}
          onEnroll={handleEnroll}
          onCancel={() => setEnrollRange(null)}
        />
      )}
    </div>
  )
}

export default App
