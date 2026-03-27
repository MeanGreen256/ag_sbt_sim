export interface Emitter {
  id: string
  name: string
  tags: string[]
  freq_range_start: number
  freq_range_end: number
  notes: string
  first_seen: string
  last_seen: string
  created_at: string
}

export interface MatchResult {
  emitter_id: string
  name: string
  confidence: number
}

export interface SubBandMatch {
  matches: MatchResult[]
  is_new: boolean
}

export interface MatchData {
  subbands: Record<string, SubBandMatch>
}

export interface Observation {
  emitter_id: string
  timestamp: string
  frequency: number
  power_db: number
  bandwidth: number
  is_active: boolean
  match_confidence: number
}

export interface Anomaly {
  type: 'power_anomaly' | 'freq_shift' | 'schedule_anomaly' | 'new_emitter'
  emitter_id: string | null
  emitter_name: string | null
  severity: number
  baseline_value: number | null
  current_value: number
  message: string
  timestamp: string
}
