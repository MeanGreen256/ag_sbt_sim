import pytest
import os
import tempfile
from datetime import datetime, timedelta
from emitter_db import EmitterDB
from models import Emitter, Fingerprint, Observation, Baseline


@pytest.fixture
def db():
    """Create a temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = EmitterDB(path)
    yield database
    database.close()
    os.unlink(path)


def test_create_emitter(db):
    e = db.create_emitter("RADAR-A", ["hostile", "radar"], 99.0, 100.0)
    assert e.name == "RADAR-A"
    assert e.tags == ["hostile", "radar"]
    assert e.id


def test_get_emitter(db):
    created = db.create_emitter("RADAR-A", [], 99.0, 100.0)
    fetched = db.get_emitter(created.id)
    assert fetched is not None
    assert fetched.name == "RADAR-A"


def test_get_emitter_not_found(db):
    assert db.get_emitter("nonexistent") is None


def test_list_emitters(db):
    db.create_emitter("A", [], 99.0, 100.0)
    db.create_emitter("B", [], 100.0, 101.0)
    emitters = db.list_emitters()
    assert len(emitters) == 2


def test_update_emitter(db):
    created = db.create_emitter("OLD", [], 99.0, 100.0)
    updated = db.update_emitter(created.id, name="NEW", tags=["updated"], notes="test note")
    assert updated.name == "NEW"
    assert updated.tags == ["updated"]
    assert updated.notes == "test note"


def test_delete_emitter(db):
    created = db.create_emitter("DEL", [], 99.0, 100.0)
    db.delete_emitter(created.id)
    assert db.get_emitter(created.id) is None


def test_add_fingerprint(db):
    e = db.create_emitter("FP-TEST", [], 99.0, 100.0)
    fp = db.add_fingerprint(e.id, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], quality_score=0.9)
    assert fp.emitter_id == e.id
    assert fp.feature_vector == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert fp.quality_score == 0.9


def test_get_all_fingerprints(db):
    e = db.create_emitter("FP-MULTI", [], 99.0, 100.0)
    db.add_fingerprint(e.id, [1.0]*7, quality_score=0.8)
    db.add_fingerprint(e.id, [2.0]*7, quality_score=0.9)
    fps = db.get_fingerprints(e.id)
    assert len(fps) == 2


def test_log_observation(db):
    e = db.create_emitter("OBS-TEST", [], 99.0, 100.0)
    db.log_observation(e.id, frequency=99.5, power_db=-45.0, bandwidth=20.0, match_confidence=0.94)
    obs = db.get_observations(e.id)
    assert len(obs) == 1
    assert obs[0].frequency == 99.5


def test_prune_observations(db):
    e = db.create_emitter("PRUNE", [], 99.0, 100.0)
    # Insert an old observation
    old_time = datetime.now() - timedelta(hours=48)
    db.log_observation(e.id, frequency=99.5, power_db=-45.0, bandwidth=20.0, timestamp=old_time)
    # Insert a recent observation
    db.log_observation(e.id, frequency=99.5, power_db=-45.0, bandwidth=20.0)
    db.prune_observations(max_age_hours=24)
    obs = db.get_observations(e.id)
    assert len(obs) == 1  # Only the recent one


def test_compute_baseline(db):
    e = db.create_emitter("BASE", [], 99.0, 100.0)
    # Add observations over the last hour
    for i in range(20):
        db.log_observation(
            e.id,
            frequency=99.5 + i * 0.001,
            power_db=-50.0 + i * 0.1,
            bandwidth=20.0,
        )
    baseline = db.compute_baseline(e.id, window_hours=4)
    assert baseline is not None
    assert baseline.power_mean != 0
    assert baseline.freq_mean > 99.0


def test_cascade_delete(db):
    e = db.create_emitter("CASCADE", [], 99.0, 100.0)
    db.add_fingerprint(e.id, [1.0]*7)
    db.log_observation(e.id, frequency=99.5, power_db=-45.0, bandwidth=20.0)
    db.delete_emitter(e.id)
    assert db.get_fingerprints(e.id) == []
    assert db.get_observations(e.id) == []
