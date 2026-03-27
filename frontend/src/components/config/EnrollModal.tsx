import { useState } from 'react'
import { Radio, X, Loader2 } from 'lucide-react'

interface Props {
  freqStart: number
  freqEnd: number
  enrolling: boolean
  onEnroll: (name: string, tags: string[]) => void
  onCancel: () => void
}

export function EnrollModal({ freqStart, freqEnd, enrolling, onEnroll, onCancel }: Props) {
  const [name, setName] = useState('')
  const [tagsInput, setTagsInput] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    const tags = tagsInput
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)
    onEnroll(name.trim(), tags)
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-80">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
            <Radio className="w-4 h-4 text-cyan-400" />
            Enroll Emitter
          </h3>
          <button onClick={onCancel} className="text-gray-500 hover:text-gray-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="text-xs text-gray-400 mb-3">
          {freqStart.toFixed(3)} – {freqEnd.toFixed(3)} MHz
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="RADAR-A"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none"
              autoFocus
              disabled={enrolling}
            />
          </div>

          <div>
            <label className="text-xs text-gray-500 block mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={tagsInput}
              onChange={e => setTagsInput(e.target.value)}
              placeholder="hostile, radar"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none"
              disabled={enrolling}
            />
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 px-3 py-1.5 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700"
              disabled={enrolling}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-3 py-1.5 text-xs rounded bg-cyan-600 text-white hover:bg-cyan-500 disabled:opacity-50 flex items-center justify-center gap-1"
              disabled={!name.trim() || enrolling}
            >
              {enrolling && <Loader2 className="w-3 h-3 animate-spin" />}
              {enrolling ? 'Capturing...' : 'Enroll'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
