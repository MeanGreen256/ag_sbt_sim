import { useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import type { SubBand } from '../../types/subband'

interface Props {
  subbands: SubBand[]
  onAdd: (sb: Omit<SubBand, 'id'>) => void
  onDelete: (id: string) => void
  pendingRange?: { start: number; end: number } | null
  onClearPending?: () => void
}

const COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#a855f7', '#ec4899']

export function SubBandEditor({ subbands, onAdd, onDelete, pendingRange, onClearPending }: Props) {
  const [name, setName] = useState('')
  const [freqStart, setFreqStart] = useState('')
  const [freqEnd, setFreqEnd] = useState('')
  const [threshold, setThreshold] = useState('-40')
  const [color, setColor] = useState(COLORS[0])

  // If there's a pending drag range, populate the form
  const effectiveStart = pendingRange ? pendingRange.start.toFixed(3) : freqStart
  const effectiveEnd = pendingRange ? pendingRange.end.toFixed(3) : freqEnd

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const start = parseFloat(pendingRange ? effectiveStart : freqStart)
    const end = parseFloat(pendingRange ? effectiveEnd : freqEnd)
    if (!name || isNaN(start) || isNaN(end)) return

    onAdd({
      name,
      freq_start: start,
      freq_end: end,
      threshold_db: parseFloat(threshold) || -40,
      color,
    })

    setName('')
    setFreqStart('')
    setFreqEnd('')
    setThreshold('-40')
    setColor(COLORS[(subbands.length + 1) % COLORS.length])
    onClearPending?.()
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Sub-Bands
      </h3>

      {/* Existing sub-bands */}
      <div className="space-y-1 mb-3">
        {subbands.length === 0 && (
          <p className="text-xs text-gray-600 italic">
            No sub-bands defined. Drag on the spectrum or add below.
          </p>
        )}
        {subbands.map(sb => (
          <div
            key={sb.id}
            className="flex items-center gap-2 text-xs bg-gray-800/50 rounded px-2 py-1"
          >
            <div
              className="w-3 h-3 rounded-sm shrink-0"
              style={{ backgroundColor: sb.color }}
            />
            <span className="flex-1 truncate">{sb.name}</span>
            <span className="text-gray-500">
              {sb.freq_start.toFixed(1)}-{sb.freq_end.toFixed(1)}
            </span>
            <span className="text-gray-500">{sb.threshold_db} dB</span>
            <button
              onClick={() => onDelete(sb.id)}
              className="text-gray-600 hover:text-red-400"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* Add form */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="flex gap-2">
          <input
            placeholder="Name"
            value={name}
            onChange={e => setName(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200"
          />
          <input
            type="color"
            value={color}
            onChange={e => setColor(e.target.value)}
            className="w-7 h-7 p-0 border-0 bg-transparent cursor-pointer"
          />
        </div>
        <div className="flex gap-2">
          <input
            placeholder="Start MHz"
            value={pendingRange ? effectiveStart : freqStart}
            onChange={e => {
              setFreqStart(e.target.value)
              onClearPending?.()
            }}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200"
          />
          <input
            placeholder="End MHz"
            value={pendingRange ? effectiveEnd : freqEnd}
            onChange={e => {
              setFreqEnd(e.target.value)
              onClearPending?.()
            }}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200"
          />
        </div>
        <div className="flex gap-2">
          <input
            placeholder="Threshold dB"
            value={threshold}
            onChange={e => setThreshold(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200"
          />
          <button
            type="submit"
            className="flex items-center gap-1 bg-blue-600 hover:bg-blue-700 text-white rounded px-3 py-1 text-xs font-medium"
          >
            <Plus className="w-3 h-3" /> Add
          </button>
        </div>
      </form>
    </div>
  )
}
