import { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import type { Emitter } from '../types/emitter'

export function useEmitters() {
  const [emitters, setEmitters] = useState<Emitter[]>([])
  const [enrolling, setEnrolling] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const list = await api.listEmitters()
      setEmitters(list)
    } catch {
      // Backend unreachable
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const enroll = useCallback(
    async (data: { freq_start: number; freq_end: number; name: string; tags: string[]; capture_frames?: number }) => {
      setEnrolling(true)
      try {
        await api.enrollEmitter(data)
        await refresh()
      } finally {
        setEnrolling(false)
      }
    },
    [refresh]
  )

  const deleteEmitter = useCallback(
    async (id: string) => {
      await api.deleteEmitter(id)
      await refresh()
    },
    [refresh]
  )

  const updateEmitter = useCallback(
    async (id: string, data: { name?: string; tags?: string[]; notes?: string }) => {
      await api.updateEmitter(id, data)
      await refresh()
    },
    [refresh]
  )

  return { emitters, enrolling, enroll, deleteEmitter, updateEmitter, refresh }
}
