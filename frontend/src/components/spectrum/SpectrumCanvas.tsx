import { useRef, useEffect, useState, useCallback } from 'react'
import type { SpectrumConfig } from '../../types/signal'
import type { SubBand } from '../../types/subband'
import { renderMatchOverlay } from './MatchOverlay'
import type { MatchData } from '../../types/emitter'

interface Props {
  spectrum: Float32Array | null
  config: SpectrumConfig
  subbands: SubBand[]
  frameCount: number
  matchData?: MatchData | null
  onSubBandDrag?: (freqStart: number, freqEnd: number) => void
}

const DB_MIN = -120
const DB_MAX = 0
const PADDING = { top: 10, right: 20, bottom: 30, left: 50 }

export function SpectrumCanvas({
  spectrum,
  config,
  subbands,
  frameCount,
  matchData,
  onSubBandDrag,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 250 })
  const [cursor, setCursor] = useState<{ x: number; freq: number; db: number } | null>(null)
  const dragStartRef = useRef<number | null>(null)
  const [dragRange, setDragRange] = useState<{ start: number; end: number } | null>(null)

  // Resize observer
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(([entry]) => {
      const { width } = entry.contentRect
      setSize({ width, height: 250 })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const freqAtX = useCallback(
    (x: number) => {
      const plotW = size.width - PADDING.left - PADDING.right
      const frac = (x - PADDING.left) / plotW
      return config.center_freq - config.bandwidth / 2 + frac * config.bandwidth
    },
    [size.width, config.center_freq, config.bandwidth]
  )

  const xAtFreq = useCallback(
    (freq: number) => {
      const plotW = size.width - PADDING.left - PADDING.right
      const frac =
        (freq - (config.center_freq - config.bandwidth / 2)) / config.bandwidth
      return PADDING.left + frac * plotW
    },
    [size.width, config.center_freq, config.bandwidth]
  )

  // Mouse handlers for crosshair and drag-to-define sub-band
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current!.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const plotH = size.height - PADDING.top - PADDING.bottom
      const dbFrac = 1 - (y - PADDING.top) / plotH
      const db = DB_MIN + dbFrac * (DB_MAX - DB_MIN)
      const freq = freqAtX(x)
      setCursor({ x, freq, db })

      if (dragStartRef.current !== null) {
        setDragRange({ start: dragStartRef.current, end: freq })
      }
    },
    [size.height, freqAtX]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current!.getBoundingClientRect()
      const x = e.clientX - rect.left
      dragStartRef.current = freqAtX(x)
      setDragRange(null)
    },
    [freqAtX]
  )

  const handleMouseUp = useCallback(() => {
    if (dragRange && onSubBandDrag) {
      const start = Math.min(dragRange.start, dragRange.end)
      const end = Math.max(dragRange.start, dragRange.end)
      if (end - start > 0.01) {
        onSubBandDrag(start, end)
      }
    }
    dragStartRef.current = null
    setDragRange(null)
  }, [dragRange, onSubBandDrag])

  const handleMouseLeave = useCallback(() => {
    setCursor(null)
    dragStartRef.current = null
    setDragRange(null)
  }, [])

  // Render
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !spectrum) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = size
    canvas.width = width
    canvas.height = height

    const plotW = width - PADDING.left - PADDING.right
    const plotH = height - PADDING.top - PADDING.bottom

    // Clear
    ctx.fillStyle = '#030712'
    ctx.fillRect(0, 0, width, height)

    // Sub-band overlays
    for (const sb of subbands) {
      const x1 = xAtFreq(sb.freq_start)
      const x2 = xAtFreq(sb.freq_end)
      ctx.fillStyle = sb.color + '30'
      ctx.fillRect(x1, PADDING.top, x2 - x1, plotH)
      ctx.strokeStyle = sb.color + '80'
      ctx.lineWidth = 1
      ctx.strokeRect(x1, PADDING.top, x2 - x1, plotH)

      // Label
      ctx.fillStyle = sb.color
      ctx.font = '10px monospace'
      ctx.fillText(sb.name, x1 + 2, PADDING.top + 12)
    }

    // Drag selection overlay
    if (dragRange) {
      const x1 = xAtFreq(Math.min(dragRange.start, dragRange.end))
      const x2 = xAtFreq(Math.max(dragRange.start, dragRange.end))
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)'
      ctx.fillRect(x1, PADDING.top, x2 - x1, plotH)
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.6)'
      ctx.setLineDash([4, 4])
      ctx.strokeRect(x1, PADDING.top, x2 - x1, plotH)
      ctx.setLineDash([])
    }

    // Grid lines
    ctx.strokeStyle = '#1f2937'
    ctx.lineWidth = 1
    const dbStep = 20
    for (let db = DB_MIN; db <= DB_MAX; db += dbStep) {
      const y = PADDING.top + plotH * (1 - (db - DB_MIN) / (DB_MAX - DB_MIN))
      ctx.beginPath()
      ctx.moveTo(PADDING.left, y)
      ctx.lineTo(PADDING.left + plotW, y)
      ctx.stroke()

      ctx.fillStyle = '#6b7280'
      ctx.font = '10px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(`${db}`, PADDING.left - 5, y + 3)
    }

    // Frequency labels
    const freqStart = config.center_freq - config.bandwidth / 2
    const numLabels = 5
    for (let i = 0; i <= numLabels; i++) {
      const freq = freqStart + (i / numLabels) * config.bandwidth
      const x = PADDING.left + (i / numLabels) * plotW
      ctx.fillStyle = '#6b7280'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`${freq.toFixed(1)}`, x, height - 5)
    }

    // Spectrum line
    ctx.beginPath()
    const step = plotW / (spectrum.length - 1)
    for (let i = 0; i < spectrum.length; i++) {
      const x = PADDING.left + i * step
      const dbNorm = (spectrum[i] - DB_MIN) / (DB_MAX - DB_MIN)
      const y = PADDING.top + plotH * (1 - Math.max(0, Math.min(1, dbNorm)))
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.strokeStyle = '#22d3ee'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Gradient fill under the line
    const gradient = ctx.createLinearGradient(0, PADDING.top, 0, PADDING.top + plotH)
    gradient.addColorStop(0, 'rgba(34, 211, 238, 0.3)')
    gradient.addColorStop(1, 'rgba(34, 211, 238, 0)')
    ctx.lineTo(PADDING.left + plotW, PADDING.top + plotH)
    ctx.lineTo(PADDING.left, PADDING.top + plotH)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Crosshair cursor
    if (cursor && cursor.x >= PADDING.left && cursor.x <= PADDING.left + plotW) {
      ctx.strokeStyle = '#9ca3af50'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      // Vertical line
      ctx.beginPath()
      ctx.moveTo(cursor.x, PADDING.top)
      ctx.lineTo(cursor.x, PADDING.top + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Readout
      ctx.fillStyle = '#e5e7eb'
      ctx.font = '11px monospace'
      ctx.textAlign = 'left'
      const label = `${cursor.freq.toFixed(3)} MHz  ${cursor.db.toFixed(1)} dB`
      ctx.fillText(label, cursor.x + 8, PADDING.top + 15)
    }

    // Axis labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Frequency (MHz)', PADDING.left + plotW / 2, height - 18)
    ctx.save()
    ctx.translate(12, PADDING.top + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Power (dBFS)', 0, 0)
    ctx.restore()

    // Match overlay badges
    renderMatchOverlay({
      ctx,
      matchData: matchData ?? null,
      subbands,
      xAtFreq,
      paddingTop: PADDING.top,
    })
  }, [spectrum, size, config, subbands, cursor, dragRange, xAtFreq, frameCount, matchData])

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full cursor-crosshair"
        style={{ height: 250 }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  )
}
