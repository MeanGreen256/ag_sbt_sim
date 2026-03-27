import { useState } from 'react'
import { Search, Trash2, Radio } from 'lucide-react'
import type { Emitter, MatchData } from '../../types/emitter'

interface Props {
  emitters: Emitter[]
  matchData: MatchData | null
  onDelete: (id: string) => void
}

export function EmitterLibrary({ emitters, matchData, onDelete }: Props) {
  const [search, setSearch] = useState('')

  const confidenceMap: Record<string, number> = {}
  if (matchData) {
    for (const sb of Object.values(matchData.subbands)) {
      for (const m of sb.matches) {
        if (!confidenceMap[m.emitter_id] || m.confidence > confidenceMap[m.emitter_id]) {
          confidenceMap[m.emitter_id] = m.confidence
        }
      }
    }
  }

  const filtered = emitters.filter(
    e =>
      e.name.toLowerCase().includes(search.toLowerCase()) ||
      e.tags.some(t => t.toLowerCase().includes(search.toLowerCase()))
  )

  const getConfidenceColor = (confidence: number | undefined) => {
    if (confidence === undefined) return 'text-gray-600'
    if (confidence >= 0.85) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceBg = (confidence: number | undefined) => {
    if (confidence === undefined) return 'bg-gray-800'
    if (confidence >= 0.85) return 'bg-green-950/30 border-green-900/30'
    if (confidence >= 0.6) return 'bg-yellow-950/30 border-yellow-900/30'
    return 'bg-red-950/30 border-red-900/30'
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <Radio className="w-3 h-3" />
        Emitter Library
      </h3>

      <div className="relative mb-2">
        <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search emitters..."
          className="w-full bg-gray-800 border border-gray-700 rounded pl-7 pr-2 py-1 text-xs text-gray-300 focus:border-cyan-500 focus:outline-none"
        />
      </div>

      <div className="space-y-1 max-h-48 overflow-y-auto">
        {filtered.length === 0 && (
          <p className="text-xs text-gray-600 italic">No emitters enrolled</p>
        )}
        {filtered.map(emitter => {
          const confidence = confidenceMap[emitter.id]
          return (
            <div
              key={emitter.id}
              className={`flex items-center gap-2 text-xs border rounded px-2 py-1 ${getConfidenceBg(confidence)}`}
            >
              <span className={`font-bold text-lg leading-none ${getConfidenceColor(confidence)}`}>
                {confidence !== undefined ? '\u25CF' : '\u25CB'}
              </span>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-gray-200 truncate">{emitter.name}</div>
                {emitter.tags.length > 0 && (
                  <div className="text-[10px] text-gray-500 truncate">
                    {emitter.tags.join(', ')}
                  </div>
                )}
              </div>
              {confidence !== undefined && (
                <span className={`text-[10px] font-mono ${getConfidenceColor(confidence)}`}>
                  {(confidence * 100).toFixed(0)}%
                </span>
              )}
              <button
                onClick={() => onDelete(emitter.id)}
                className="text-gray-600 hover:text-red-400 shrink-0"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
