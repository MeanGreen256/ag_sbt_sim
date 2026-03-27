import { useRef, useEffect, useState } from 'react'
import { VIRIDIS_LUT } from '../../utils/colormap'
import type { RingBuffer } from '../../utils/ringBuffer'

interface Props {
  waterfallBuffer: RingBuffer | null
  frameCount: number
}

const DB_MIN = -120
const DB_MAX = 0
const HEIGHT = 200

export function WaterfallCanvas({ waterfallBuffer, frameCount }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(([entry]) => {
      setWidth(Math.floor(entry.contentRect.width))
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !waterfallBuffer || waterfallBuffer.count === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = HEIGHT

    const imageData = ctx.createImageData(width, HEIGHT)
    const pixels = imageData.data
    const bins = waterfallBuffer.width

    for (let row = 0; row < HEIGHT; row++) {
      // Map display row to ring buffer age
      const age = Math.floor((row / HEIGHT) * Math.min(waterfallBuffer.count, waterfallBuffer.rows))
      const spectrum = waterfallBuffer.getRow(age)

      for (let col = 0; col < width; col++) {
        // Map display column to FFT bin
        const bin = Math.floor((col / width) * bins)
        const db = spectrum[bin]

        // Normalize to 0-255 for colormap lookup
        const norm = Math.max(0, Math.min(255, Math.round(((db - DB_MIN) / (DB_MAX - DB_MIN)) * 255)))
        const lutIdx = norm * 3

        const pixIdx = (row * width + col) * 4
        pixels[pixIdx] = VIRIDIS_LUT[lutIdx]
        pixels[pixIdx + 1] = VIRIDIS_LUT[lutIdx + 1]
        pixels[pixIdx + 2] = VIRIDIS_LUT[lutIdx + 2]
        pixels[pixIdx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [waterfallBuffer, width, frameCount])

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: HEIGHT }}
      />
    </div>
  )
}
