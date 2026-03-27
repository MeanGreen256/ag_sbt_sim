export interface SubBand {
  id: string
  name: string
  freq_start: number // MHz
  freq_end: number // MHz
  color: string
  threshold_db: number
}

export interface Alert {
  id: string
  subband_id: string
  subband_name: string
  timestamp: string
  power_db: number
  threshold_db: number
}
