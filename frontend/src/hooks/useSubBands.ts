import { useState, useEffect, useCallback } from 'react'
import type { SubBand } from '../types/subband'
import { api } from '../services/api'

export function useSubBands() {
  const [subbands, setSubBands] = useState<SubBand[]>([])

  const refresh = useCallback(async () => {
    try {
      const data = await api.listSubBands()
      setSubBands(data)
    } catch {
      // Backend may not be ready yet — silently ignore
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const addSubBand = useCallback(
    async (sb: Omit<SubBand, 'id'>) => {
      await api.createSubBand(sb)
      await refresh()
    },
    [refresh]
  )

  const updateSubBand = useCallback(
    async (id: string, sb: SubBand) => {
      await api.updateSubBand(id, sb)
      await refresh()
    },
    [refresh]
  )

  const deleteSubBand = useCallback(
    async (id: string) => {
      await api.deleteSubBand(id)
      await refresh()
    },
    [refresh]
  )

  return { subbands, addSubBand, updateSubBand, deleteSubBand }
}
