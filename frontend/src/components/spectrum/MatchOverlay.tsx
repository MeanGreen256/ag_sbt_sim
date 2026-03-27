import type { MatchData } from '../../types/emitter'
import type { SubBand } from '../../types/subband'

interface OverlayParams {
  ctx: CanvasRenderingContext2D
  matchData: MatchData | null
  subbands: SubBand[]
  xAtFreq: (freq: number) => number
  paddingTop: number
}

export function renderMatchOverlay({ ctx, matchData, subbands, xAtFreq, paddingTop }: OverlayParams) {
  if (!matchData) return

  for (const sb of subbands) {
    const sbMatch = matchData.subbands[sb.id]
    if (!sbMatch || sbMatch.matches.length === 0) continue

    const best = sbMatch.matches[0]
    const x = xAtFreq((sb.freq_start + sb.freq_end) / 2)
    const y = paddingTop + 28

    let bgColor: string
    let textColor: string
    if (best.confidence >= 0.85) {
      bgColor = 'rgba(34, 197, 94, 0.8)'
      textColor = '#ffffff'
    } else if (best.confidence >= 0.6) {
      bgColor = 'rgba(234, 179, 8, 0.8)'
      textColor = '#000000'
    } else {
      bgColor = 'rgba(239, 68, 68, 0.8)'
      textColor = '#ffffff'
    }

    const label = `${best.name} ${(best.confidence * 100).toFixed(0)}%`
    ctx.font = '10px monospace'
    const metrics = ctx.measureText(label)
    const badgeW = metrics.width + 8
    const badgeH = 14

    ctx.fillStyle = bgColor
    ctx.beginPath()
    ctx.roundRect(x - badgeW / 2, y - badgeH / 2, badgeW, badgeH, 3)
    ctx.fill()

    ctx.fillStyle = textColor
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, x, y)

    if (sbMatch.is_new) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'
      ctx.font = 'bold 9px monospace'
      ctx.fillText('NEW', x, y + 14)
    }
  }

  ctx.textAlign = 'start'
  ctx.textBaseline = 'alphabetic'
}
