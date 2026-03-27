from models import Observation, Anomaly
from emitter_db import EmitterDB


class AnomalyDetector:
    def __init__(self, db: EmitterDB, power_sigma_threshold: float = 2.0,
                 freq_shift_threshold_khz: float = 5.0, new_emitter_confidence_threshold: float = 0.6):
        self.db = db
        self.power_sigma_threshold = power_sigma_threshold
        self.freq_shift_threshold_khz = freq_shift_threshold_khz
        self.new_emitter_confidence_threshold = new_emitter_confidence_threshold

    def check(self, emitter_id: str, current: Observation) -> list[Anomaly]:
        baseline = self.db.get_baseline(emitter_id)
        if baseline is None:
            return []
        anomalies: list[Anomaly] = []
        emitter = self.db.get_emitter(emitter_id)
        emitter_name = emitter.name if emitter else None

        # Power anomaly
        if baseline.power_std > 0:
            deviation = abs(current.power_db - baseline.power_mean)
            sigma_ratio = deviation / baseline.power_std
            if sigma_ratio > self.power_sigma_threshold:
                anomalies.append(Anomaly(type="power_anomaly", emitter_id=emitter_id, emitter_name=emitter_name,
                    severity=round(sigma_ratio, 2), baseline_value=baseline.power_mean, current_value=current.power_db,
                    message=f"Power {current.power_db:.1f} dBm deviates {sigma_ratio:.1f}σ from baseline {baseline.power_mean:.1f} dBm"))
        else:
            if abs(current.power_db - baseline.power_mean) > 1.0:
                anomalies.append(Anomaly(type="power_anomaly", emitter_id=emitter_id, emitter_name=emitter_name,
                    severity=abs(current.power_db - baseline.power_mean), baseline_value=baseline.power_mean,
                    current_value=current.power_db,
                    message=f"Power {current.power_db:.1f} dBm deviates from baseline {baseline.power_mean:.1f} dBm (zero variance baseline)"))

        # Frequency shift
        freq_shift_mhz = abs(current.frequency - baseline.freq_mean)
        freq_shift_khz = freq_shift_mhz * 1000.0
        if freq_shift_khz > self.freq_shift_threshold_khz:
            anomalies.append(Anomaly(type="freq_shift", emitter_id=emitter_id, emitter_name=emitter_name,
                severity=round(freq_shift_khz, 2), baseline_value=baseline.freq_mean, current_value=current.frequency,
                message=f"Frequency shifted {freq_shift_khz:.1f} kHz from baseline {baseline.freq_mean:.3f} MHz"))

        # Schedule anomaly
        if baseline.typical_hours:
            current_hour = current.timestamp.hour
            in_typical = any(start <= current_hour < end for start, end in baseline.typical_hours)
            if not in_typical and current.is_active:
                anomalies.append(Anomaly(type="schedule_anomaly", emitter_id=emitter_id, emitter_name=emitter_name,
                    severity=1.0, current_value=float(current_hour),
                    message=f"Active at hour {current_hour}, outside typical schedule"))

        return anomalies

    def check_new_emitter(self, frequency: float, power_db: float, confidence: float) -> Anomaly | None:
        if confidence >= self.new_emitter_confidence_threshold:
            return None
        return Anomaly(type="new_emitter", emitter_id=None, emitter_name=None,
            severity=round(confidence, 4), current_value=power_db,
            message=f"Unknown emitter at {frequency:.3f} MHz, best match confidence {confidence:.0%}")
