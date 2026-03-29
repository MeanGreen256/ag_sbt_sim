from unittest.mock import patch
from alert_engine import AlertEngine
from models import SubBand, Anomaly


def make_subband(threshold_db=-20.0, **kwargs):
    defaults = dict(name="test-sb", freq_start=99.0, freq_end=100.0, threshold_db=threshold_db)
    defaults.update(kwargs)
    return SubBand(**defaults)


def test_no_alert_below_threshold():
    engine = AlertEngine()
    sb = make_subband(threshold_db=-20.0)
    alerts = engine.check([sb], {sb.id: -30.0})
    assert alerts == []


def test_alert_fires_above_threshold():
    engine = AlertEngine()
    sb = make_subband(threshold_db=-20.0)
    alerts = engine.check([sb], {sb.id: -10.0})
    assert len(alerts) == 1
    assert alerts[0].subband_id == sb.id
    assert alerts[0].subband_name == sb.name
    assert alerts[0].power_db == -10.0
    assert alerts[0].threshold_db == -20.0


def test_alert_debounce():
    engine = AlertEngine(debounce_seconds=5.0)
    sb = make_subband(threshold_db=-20.0)

    # First alert fires
    alerts1 = engine.check([sb], {sb.id: -10.0})
    assert len(alerts1) == 1

    # Disarm by dropping power well below threshold - hysteresis
    engine.check([sb], {sb.id: -50.0})

    # Second crossing within debounce window — should NOT fire
    alerts2 = engine.check([sb], {sb.id: -10.0})
    assert len(alerts2) == 0


@patch("alert_engine.time")
def test_alert_fires_after_debounce(mock_time):
    mock_time.time.return_value = 1000.0
    engine = AlertEngine(debounce_seconds=2.0)
    sb = make_subband(threshold_db=-20.0)

    # First alert at t=1000
    alerts1 = engine.check([sb], {sb.id: -10.0})
    assert len(alerts1) == 1

    # Disarm
    engine.check([sb], {sb.id: -50.0})

    # Advance time past debounce
    mock_time.time.return_value = 1003.0
    alerts2 = engine.check([sb], {sb.id: -10.0})
    assert len(alerts2) == 1


def test_hysteresis_stays_armed():
    engine = AlertEngine(hysteresis_db=5.0)
    sb = make_subband(threshold_db=-20.0)

    # Trigger alert
    engine.check([sb], {sb.id: -10.0})
    assert engine._armed[sb.id] is True

    # Power drops to threshold - 3 dB (within hysteresis band) — stays armed
    engine.check([sb], {sb.id: -23.0})
    assert engine._armed[sb.id] is True


def test_hysteresis_disarms():
    engine = AlertEngine(hysteresis_db=5.0)
    sb = make_subband(threshold_db=-20.0)

    # Trigger
    engine.check([sb], {sb.id: -10.0})
    assert engine._armed[sb.id] is True

    # Drop below threshold - hysteresis (-25 dB)
    engine.check([sb], {sb.id: -26.0})
    assert engine._armed[sb.id] is False


@patch("alert_engine.time")
def test_multiple_subbands_independent(mock_time):
    mock_time.time.return_value = 1000.0
    engine = AlertEngine(debounce_seconds=0.0)
    sb1 = make_subband(threshold_db=-20.0, name="sb1")
    sb2 = make_subband(threshold_db=-30.0, name="sb2")

    # Only sb1 exceeds its threshold
    alerts = engine.check([sb1, sb2], {sb1.id: -10.0, sb2.id: -35.0})
    assert len(alerts) == 1
    assert alerts[0].subband_name == "sb1"


def test_missing_subband_power_skipped():
    engine = AlertEngine()
    sb = make_subband(threshold_db=-20.0)
    # Empty power dict — no crash, no alerts
    alerts = engine.check([sb], {})
    assert alerts == []


def test_format_anomaly_alert_mapping():
    engine = AlertEngine()
    anomaly = Anomaly(
        type="power_anomaly",
        emitter_id="em-123",
        emitter_name="RADAR-A",
        severity=3.5,
        baseline_value=-40.0,
        current_value=-20.0,
        message="Power spike",
    )
    alert = engine.format_anomaly_alert(anomaly)
    assert alert.subband_id == "em-123"
    assert "[ANOMALY]" in alert.subband_name
    assert "RADAR-A" in alert.subband_name
    assert alert.power_db == -20.0
    assert alert.threshold_db == -40.0


def test_format_anomaly_alert_none_fields():
    engine = AlertEngine()
    anomaly = Anomaly(
        type="new_emitter",
        emitter_id=None,
        emitter_name=None,
        severity=0.5,
        baseline_value=None,
        current_value=-30.0,
        message="Unknown emitter",
    )
    alert = engine.format_anomaly_alert(anomaly)
    assert alert.subband_id == "unknown"
    assert "Unknown" in alert.subband_name
    assert alert.threshold_db == 0.0
