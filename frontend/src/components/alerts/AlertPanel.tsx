import { Bell } from 'lucide-react'
import type { Alert } from '../../types/subband'

interface Props {
  alerts: Alert[]
}

export function AlertPanel({ alerts }: Props) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Alerts
        </h3>
        {alerts.length > 0 && (
          <span className="bg-red-600 text-white text-[10px] font-bold rounded-full px-1.5 py-0.5 leading-none">
            {alerts.length}
          </span>
        )}
      </div>

      <div className="space-y-1 max-h-48 overflow-y-auto">
        {alerts.length === 0 && (
          <p className="text-xs text-gray-600 italic flex items-center gap-1.5">
            <Bell className="w-3 h-3" /> No alerts
          </p>
        )}
        {[...alerts].reverse().map(alert => (
          <div
            key={alert.id}
            className="flex items-center gap-2 text-xs bg-red-950/30 border border-red-900/30 rounded px-2 py-1"
          >
            <Bell className="w-3 h-3 text-red-400 shrink-0" />
            <span className="font-medium text-red-300">{alert.subband_name}</span>
            <span className="text-gray-500">
              {alert.power_db.toFixed(1)} dB &gt; {alert.threshold_db} dB
            </span>
            <span className="text-gray-600 ml-auto text-[10px]">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
