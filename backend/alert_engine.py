import time
from models import SubBand, Alert


class AlertEngine:
    """Monitors sub-band power levels and generates alerts with hysteresis and debounce."""

    def __init__(self, hysteresis_db: float = 5.0, debounce_seconds: float = 2.0):
        self.hysteresis_db = hysteresis_db
        self.debounce_seconds = debounce_seconds

        # Track armed state per sub-band: True = currently in alert
        self._armed: dict[str, bool] = {}
        # Last alert timestamp per sub-band
        self._last_alert_time: dict[str, float] = {}

    def check(
        self, subbands: list[SubBand], subband_powers: dict[str, float]
    ) -> list[Alert]:
        """Check sub-band power levels against thresholds.

        Args:
            subbands: List of configured sub-bands
            subband_powers: Dict mapping subband id to measured power in dB

        Returns:
            List of new Alert objects triggered this cycle
        """
        alerts: list[Alert] = []
        now = time.time()

        for sb in subbands:
            power = subband_powers.get(sb.id)
            if power is None:
                continue

            was_armed = self._armed.get(sb.id, False)

            if not was_armed:
                # Trigger if power exceeds threshold
                if power > sb.threshold_db:
                    self._armed[sb.id] = True
                    # Check debounce
                    last = self._last_alert_time.get(sb.id, 0)
                    if now - last >= self.debounce_seconds:
                        self._last_alert_time[sb.id] = now
                        alerts.append(
                            Alert(
                                subband_id=sb.id,
                                subband_name=sb.name,
                                power_db=round(power, 1),
                                threshold_db=sb.threshold_db,
                            )
                        )
            else:
                # Disarm only if power drops below threshold - hysteresis
                if power < sb.threshold_db - self.hysteresis_db:
                    self._armed[sb.id] = False

        return alerts
