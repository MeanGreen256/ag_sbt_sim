import type { Emitter, MatchData } from '../../types/emitter'

interface Props {
  emitters: Emitter[]
  matchData: MatchData | null
}

export function EmitterTimeline({ emitters, matchData }: Props) {
  if (emitters.length === 0) return null

  const activeEmitters = new Set<string>()
  if (matchData) {
    for (const sb of Object.values(matchData.subbands)) {
      for (const m of sb.matches) {
        if (m.confidence >= 0.6) {
          activeEmitters.add(m.emitter_id)
        }
      }
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-2">
      <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">
        Emitter Activity
      </h3>
      <div className="space-y-0.5">
        {emitters.map(emitter => {
          const active = activeEmitters.has(emitter.id)
          return (
            <div key={emitter.id} className="flex items-center gap-2 text-[10px]">
              <span className="text-gray-400 w-20 truncate font-mono">{emitter.name}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    active ? 'bg-cyan-500' : 'bg-gray-700'
                  }`}
                  style={{ width: active ? '100%' : '0%' }}
                />
              </div>
              <span className={`w-6 text-right ${active ? 'text-cyan-400' : 'text-gray-600'}`}>
                {active ? 'ON' : 'OFF'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
