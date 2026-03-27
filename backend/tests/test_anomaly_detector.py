import pytest
import os
import tempfile
from datetime import datetime
from emitter_db import EmitterDB
from anomaly_detector import AnomalyDetector
from models import Observation, Baseline


@pytest.fixture
def setup():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    detector = AnomalyDetector(db)
    emitter = db.create_emitter("TEST-E", [], 99.0, 100.0)
    for i in range(20):
        db.log_observation(emitter.id, frequency=99.5, power_db=-50.0, bandwidth=20.0)
    db.compute_baseline(emitter.id)
    yield detector, db, emitter
    db.close()
    os.unlink(path)


def test_no_anomaly_within_baseline(setup):
    detector, db, emitter = setup
    obs = Observation(emitter_id=emitter.id, frequency=99.5, power_db=-50.0, bandwidth=20.0, match_confidence=0.95)
    anomalies = detector.check(emitter.id, obs)
    assert len(anomalies) == 0


def test_power_anomaly(setup):
    detector, db, emitter = setup
    obs = Observation(emitter_id=emitter.id, frequency=99.5, power_db=-30.0, bandwidth=20.0, match_confidence=0.95)
    anomalies = detector.check(emitter.id, obs)
    types = [a.type for a in anomalies]
    assert "power_anomaly" in types


def test_freq_shift_anomaly(setup):
    detector, db, emitter = setup
    obs = Observation(emitter_id=emitter.id, frequency=99.510, power_db=-50.0, bandwidth=20.0, match_confidence=0.95)
    anomalies = detector.check(emitter.id, obs)
    types = [a.type for a in anomalies]
    assert "freq_shift" in types


def test_no_anomaly_small_freq_shift(setup):
    detector, db, emitter = setup
    obs = Observation(emitter_id=emitter.id, frequency=99.502, power_db=-50.0, bandwidth=20.0, match_confidence=0.95)
    anomalies = detector.check(emitter.id, obs)
    freq_anomalies = [a for a in anomalies if a.type == "freq_shift"]
    assert len(freq_anomalies) == 0


def test_new_emitter_anomaly():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    detector = AnomalyDetector(db)
    anomaly = detector.check_new_emitter(frequency=99.5, power_db=-40.0, confidence=0.3)
    assert anomaly is not None
    assert anomaly.type == "new_emitter"
    assert anomaly.severity == 0.3
    db.close()
    os.unlink(path)


def test_no_new_emitter_anomaly_high_confidence():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    detector = AnomalyDetector(db)
    anomaly = detector.check_new_emitter(frequency=99.5, power_db=-40.0, confidence=0.9)
    assert anomaly is None
    db.close()
    os.unlink(path)


def test_no_baseline_no_anomaly(setup):
    detector, db, emitter = setup
    new_e = db.create_emitter("NEW", [], 100.0, 101.0)
    obs = Observation(emitter_id=new_e.id, frequency=100.5, power_db=-40.0, bandwidth=20.0)
    anomalies = detector.check(new_e.id, obs)
    assert len(anomalies) == 0
