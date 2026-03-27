import type { SpectrumConfig } from '../types/signal'
import type { SubBand, Alert } from '../types/subband'

const BASE = '/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export const api = {
  // Config
  getConfig: () => fetch(`${BASE}/config`).then(r => json<SpectrumConfig>(r)),
  updateConfig: (config: SpectrumConfig) =>
    fetch(`${BASE}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }).then(r => json<SpectrumConfig>(r)),

  // Sub-bands
  listSubBands: () => fetch(`${BASE}/subbands`).then(r => json<SubBand[]>(r)),
  createSubBand: (sb: Omit<SubBand, 'id'>) =>
    fetch(`${BASE}/subbands`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sb),
    }).then(r => json<SubBand>(r)),
  updateSubBand: (id: string, sb: SubBand) =>
    fetch(`${BASE}/subbands/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sb),
    }).then(r => json<SubBand>(r)),
  deleteSubBand: (id: string) =>
    fetch(`${BASE}/subbands/${id}`, { method: 'DELETE' }),

  // Alerts
  listAlerts: () => fetch(`${BASE}/alerts`).then(r => json<Alert[]>(r)),
}
