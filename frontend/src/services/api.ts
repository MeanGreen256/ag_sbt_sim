import type { SpectrumConfig } from '../types/signal'
import type { SubBand, Alert } from '../types/subband'
import type { Emitter, Observation } from '../types/emitter'

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

  // Emitters
  listEmitters: () => fetch(`${BASE}/emitters`).then(r => json<Emitter[]>(r)),
  enrollEmitter: (data: { freq_start: number; freq_end: number; name: string; tags: string[]; capture_frames?: number }) =>
    fetch(`${BASE}/emitters/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(r => json<Emitter>(r)),
  getEmitter: (id: string) =>
    fetch(`${BASE}/emitters/${id}`).then(r => json<{ emitter: Emitter; fingerprints: unknown[]; baseline: unknown }>(r)),
  updateEmitter: (id: string, data: { name?: string; tags?: string[]; notes?: string }) =>
    fetch(`${BASE}/emitters/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(r => json<Emitter>(r)),
  deleteEmitter: (id: string) =>
    fetch(`${BASE}/emitters/${id}`, { method: 'DELETE' }),
  getEmitterHistory: (id: string, since?: string, until?: string) => {
    const params = new URLSearchParams()
    if (since) params.set('since', since)
    if (until) params.set('until', until)
    const qs = params.toString()
    return fetch(`${BASE}/emitters/${id}/history${qs ? '?' + qs : ''}`).then(r => json<Observation[]>(r))
  },
  enrichEmitter: (id: string, data: { freq_start: number; freq_end: number; capture_frames?: number }) =>
    fetch(`${BASE}/emitters/${id}/enrich`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(r => json<Emitter>(r)),
}
