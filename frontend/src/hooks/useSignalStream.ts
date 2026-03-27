import { useEffect, useRef, useState } from 'react'
import { MESSAGE_TYPE } from '../types/signal'
import { RingBuffer } from '../utils/ringBuffer'
import type { Alert } from '../types/subband'

const WATERFALL_ROWS = 200
const RECONNECT_BASE_MS = 500
const RECONNECT_MAX_MS = 10000

interface SignalStreamState {
  currentSpectrum: Float32Array | null
  waterfallBuffer: RingBuffer | null
  connected: boolean
  alerts: Alert[]
  frameCount: number
}

export function useSignalStream(fftSize: number, streaming: boolean) {
  const [state, setState] = useState<SignalStreamState>({
    currentSpectrum: null,
    waterfallBuffer: null,
    connected: false,
    alerts: [],
    frameCount: 0,
  })

  const bufferRef = useRef<RingBuffer | null>(null)
  const frameCountRef = useRef(0)
  const alertsRef = useRef<Alert[]>([])

  useEffect(() => {
    if (!streaming) return

    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined
    let reconnectDelay = RECONNECT_BASE_MS
    let cancelled = false

    function connect() {
      if (cancelled) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${protocol}//${window.location.host}/ws/spectrum`)
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        if (cancelled) { ws?.close(); return }
        reconnectDelay = RECONNECT_BASE_MS
        setState(s => ({ ...s, connected: true }))
      }

      ws.onmessage = (event) => {
        if (cancelled) return
        const data = event.data as ArrayBuffer
        if (data.byteLength < 2) return

        const view = new Uint8Array(data)
        const msgType = view[0]

        if (msgType === MESSAGE_TYPE.SPECTRUM) {
          const spectrum = new Float32Array(data.slice(1))

          if (!bufferRef.current || bufferRef.current.width !== spectrum.length) {
            bufferRef.current = new RingBuffer(spectrum.length, WATERFALL_ROWS)
          }
          bufferRef.current.push(spectrum)
          frameCountRef.current++

          setState(s => ({
            ...s,
            currentSpectrum: spectrum,
            waterfallBuffer: bufferRef.current,
            frameCount: frameCountRef.current,
          }))
        } else if (msgType === MESSAGE_TYPE.ALERT) {
          const jsonStr = new TextDecoder().decode(data.slice(1))
          const alert: Alert = JSON.parse(jsonStr)
          alertsRef.current = [...alertsRef.current.slice(-99), alert]
          setState(s => ({ ...s, alerts: alertsRef.current }))
        }
      }

      ws.onclose = () => {
        setState(s => ({ ...s, connected: false }))
        ws = null
        if (!cancelled) {
          reconnectTimer = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS)
            connect()
          }, reconnectDelay)
        }
      }

      ws.onerror = () => {
        ws?.close()
      }
    }

    connect()

    return () => {
      cancelled = true
      clearTimeout(reconnectTimer)
      if (ws) {
        ws.onclose = null // prevent reconnect on cleanup close
        ws.close()
      }
    }
  }, [streaming])

  // Reset buffer when FFT size changes
  useEffect(() => {
    bufferRef.current = null
  }, [fftSize])

  return state
}
