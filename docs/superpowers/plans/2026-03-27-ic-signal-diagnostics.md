# IC Signal Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the RF Sub-Band Signal Monitor with emitter fingerprinting, live library matching, behavioral tracking, and anomaly detection.

**Architecture:** Feature extraction pipeline computes 7-dimensional vectors from spectrum data. SQLite stores emitter fingerprints and observations. Cosine similarity matches live signals against the library every frame. Behavioral baselines enable anomaly detection (power, frequency, schedule). Results stream to frontend via a new 0x03 WebSocket message type.

**Tech Stack:** Python (NumPy, SciPy, SQLite3, FastAPI, Pydantic), React 19 + TypeScript + Tailwind v4 + Lucide icons. No new dependencies.

**Design Spec:** `docs/superpowers/specs/2026-03-27-ic-signal-diagnostics-design.md`

---

## File Map

### New Backend Files

| File | Responsibility |
|------|---------------|
| `backend/feature_extractor.py` | `FeatureExtractor` — computes 7-dim feature vectors from spectrum arrays |
| `backend/emitter_db.py` | `EmitterDB` — SQLite schema, CRUD, observation logging, baseline computation, retention pruning |
| `backend/emitter_matcher.py` | `EmitterMatcher` — cosine similarity matching, enrollment, library management |
| `backend/anomaly_detector.py` | `AnomalyDetector` — checks observations against baselines, generates anomaly alerts |
| `backend/tests/test_feature_extractor.py` | Tests for feature extraction |
| `backend/tests/test_emitter_db.py` | Tests for SQLite operations |
| `backend/tests/test_emitter_matcher.py` | Tests for matching engine |
| `backend/tests/test_anomaly_detector.py` | Tests for anomaly detection |
| `backend/tests/test_api_emitters.py` | Tests for new REST endpoints |

### Modified Backend Files

| File | Changes |
|------|---------|
| `backend/models.py` | Add `FeatureVector`, `Emitter`, `Fingerprint`, `MatchResult`, `Observation`, `Baseline`, `Anomaly` models; add `MessageType.MATCH = 0x03` |
| `backend/main.py` | New REST endpoints for emitter CRUD/enroll/enrich/history; integrate matcher + anomaly detector into WebSocket loop; emit 0x03 messages |
| `backend/alert_engine.py` | Accept `Anomaly` objects alongside threshold alerts |

### New Frontend Files

| File | Responsibility |
|------|---------------|
| `frontend/src/types/emitter.ts` | TypeScript interfaces: `Emitter`, `MatchResult`, `Observation`, `Anomaly`, `SubBandMatch` |
| `frontend/src/hooks/useEmitters.ts` | Emitter CRUD + enrollment API calls |
| `frontend/src/hooks/useMatchStream.ts` | Parse 0x03 WebSocket messages into per-subband match state |
| `frontend/src/components/config/EnrollModal.tsx` | Modal for naming/tagging emitter during enrollment |
| `frontend/src/components/emitters/EmitterLibrary.tsx` | Sidebar panel: searchable list of known emitters with match confidence |
| `frontend/src/components/spectrum/MatchOverlay.tsx` | Emitter name + confidence badge rendered on spectrum canvas |
| `frontend/src/components/emitters/EmitterTimeline.tsx` | Horizontal activity timeline below waterfall |

### Modified Frontend Files

| File | Changes |
|------|---------|
| `frontend/src/types/signal.ts` | Add `MESSAGE_TYPE.MATCH = 0x03` |
| `frontend/src/hooks/useSignalStream.ts` | Parse 0x03 message type, expose match data |
| `frontend/src/services/api.ts` | Add emitter CRUD, enroll, enrich, history API calls |
| `frontend/src/components/spectrum/SpectrumCanvas.tsx` | Integrate `MatchOverlay`, modify drag to offer enroll vs sub-band |
| `frontend/src/components/alerts/AlertPanel.tsx` | Distinguish anomaly alerts (amber) from threshold alerts (red) |
| `frontend/src/App.tsx` | Add `EmitterLibrary` to sidebar, `EmitterTimeline` below waterfall, wire up match data |

---

## Task 1: Pydantic Models

**Files:**
- Modify: `backend/models.py`
- Test: `backend/tests/test_models.py`

- [ ] **Step 1: Create test directory and write model tests**

```bash
mkdir -p backend/tests && touch backend/tests/__init__.py
```

```python
# backend/tests/test_models.py
import numpy as np
from models import FeatureVector, Emitter, MatchResult, Observation, Anomaly, Baseline, MessageType


def test_feature_vector_from_array():
    fv = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert len(fv.values) == 7
    assert fv.spectral_centroid == 1.0
    assert fv.bandwidth_3db == 2.0
    assert fv.bandwidth_10db == 3.0
    assert fv.snr == 4.0
    assert fv.papr == 5.0
    assert fv.spectral_flatness == 6.0
    assert fv.kurtosis == 7.0


def test_feature_vector_to_numpy():
    fv = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    arr = fv.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (7,)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_feature_vector_validation_rejects_wrong_length():
    import pytest
    with pytest.raises(Exception):
        FeatureVector(values=[1.0, 2.0])


def test_emitter_defaults():
    e = Emitter(name="RADAR-A", tags=["hostile"], freq_range_start=99.0, freq_range_end=100.0)
    assert e.id  # auto-generated
    assert e.name == "RADAR-A"
    assert e.tags == ["hostile"]
    assert e.notes == ""


def test_match_result_ordering():
    m1 = MatchResult(emitter_id="a", emitter_name="A", confidence=0.94)
    m2 = MatchResult(emitter_id="b", emitter_name="B", confidence=0.61)
    results = sorted([m2, m1], key=lambda r: r.confidence, reverse=True)
    assert results[0].confidence == 0.94


def test_anomaly_model():
    a = Anomaly(
        type="power_anomaly",
        emitter_id="abc",
        emitter_name="RADAR-A",
        severity=2.5,
        baseline_value=-50.0,
        current_value=-42.0,
        message="Power +8dB above baseline",
    )
    assert a.type == "power_anomaly"
    assert a.timestamp  # auto-set


def test_message_type_match():
    assert MessageType.MATCH == 0x03
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_models.py -v
```

Expected: FAIL — `FeatureVector`, `Emitter`, `MatchResult`, `Observation`, `Anomaly` not defined in `models.py`.

- [ ] **Step 3: Add new models to models.py**

Add the following to the end of `backend/models.py`:

```python
import numpy as np


class FeatureVector(BaseModel):
    """7-dimensional feature vector for signal fingerprinting."""
    values: list[float] = Field(..., min_length=7, max_length=7)

    @property
    def spectral_centroid(self) -> float:
        return self.values[0]

    @property
    def bandwidth_3db(self) -> float:
        return self.values[1]

    @property
    def bandwidth_10db(self) -> float:
        return self.values[2]

    @property
    def snr(self) -> float:
        return self.values[3]

    @property
    def papr(self) -> float:
        return self.values[4]

    @property
    def spectral_flatness(self) -> float:
        return self.values[5]

    @property
    def kurtosis(self) -> float:
        return self.values[6]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float64)


class Emitter(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    tags: list[str] = Field(default_factory=list)
    freq_range_start: float  # MHz
    freq_range_end: float  # MHz
    notes: str = ""
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)


class Fingerprint(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    emitter_id: str
    feature_vector: list[float]  # JSON-serializable 7-element array
    quality_score: float = 1.0
    captured_at: datetime = Field(default_factory=datetime.now)


class MatchResult(BaseModel):
    emitter_id: str
    emitter_name: str
    confidence: float  # 0.0 to 1.0


class Observation(BaseModel):
    emitter_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    frequency: float  # centroid MHz
    power_db: float
    bandwidth: float  # 3dB bandwidth kHz
    is_active: bool = True
    match_confidence: float = 0.0


class Baseline(BaseModel):
    emitter_id: str
    power_mean: float = 0.0
    power_std: float = 0.0
    freq_mean: float = 0.0
    freq_std: float = 0.0
    typical_hours: list[list[int]] = Field(default_factory=list)  # [[start_h, end_h], ...]
    duty_on_seconds: float = 0.0
    duty_off_seconds: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.now)


class Anomaly(BaseModel):
    type: str  # power_anomaly | freq_shift | schedule_anomaly | new_emitter
    emitter_id: str | None = None
    emitter_name: str | None = None
    severity: float = 0.0
    baseline_value: float | None = None
    current_value: float = 0.0
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
```

Also add `MATCH = 0x03` to the `MessageType` class:

```python
class MessageType:
    SPECTRUM = 0x01
    ALERT = 0x02
    MATCH = 0x03
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_models.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models.py backend/tests/
git commit -m "feat: add Pydantic models for emitter fingerprinting and anomaly detection"
```

---

## Task 2: Feature Extractor

**Files:**
- Create: `backend/feature_extractor.py`
- Test: `backend/tests/test_feature_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_feature_extractor.py
import numpy as np
import pytest
from feature_extractor import FeatureExtractor
from models import FeatureVector


@pytest.fixture
def extractor():
    return FeatureExtractor()


def make_tone_spectrum(fft_size=1024, tone_bin=512, tone_power_db=-10.0, noise_floor_db=-80.0):
    """Create a synthetic spectrum with a single tone."""
    spectrum = np.full(fft_size, noise_floor_db, dtype=np.float64)
    # Add a tone with some width (3 bins at -10, neighbors at -13)
    spectrum[tone_bin] = tone_power_db
    spectrum[tone_bin - 1] = tone_power_db - 3
    spectrum[tone_bin + 1] = tone_power_db - 3
    spectrum[tone_bin - 2] = tone_power_db - 10
    spectrum[tone_bin + 2] = tone_power_db - 10
    return spectrum


def test_extract_returns_feature_vector(extractor):
    spectrum = make_tone_spectrum()
    fv = extractor.extract(spectrum, 500, 524)
    assert isinstance(fv, FeatureVector)
    assert len(fv.values) == 7


def test_extract_spectral_centroid_near_peak(extractor):
    spectrum = make_tone_spectrum(tone_bin=512)
    fv = extractor.extract(spectrum, 500, 524)
    # Centroid should be near bin 512, which is ~0.5 normalized position
    assert 0.4 < fv.spectral_centroid < 0.6


def test_extract_snr_positive_for_tone(extractor):
    spectrum = make_tone_spectrum(tone_power_db=-10.0, noise_floor_db=-80.0)
    fv = extractor.extract(spectrum, 500, 524)
    assert fv.snr > 30  # ~70dB tone above noise, but subband SNR is lower


def test_extract_bandwidth_3db(extractor):
    spectrum = make_tone_spectrum()
    fv = extractor.extract(spectrum, 500, 524)
    assert fv.bandwidth_3db > 0  # Should have nonzero bandwidth


def test_extract_spectral_flatness_low_for_tone(extractor):
    spectrum = make_tone_spectrum()
    fv = extractor.extract(spectrum, 500, 524)
    # Tone-like signal should have low spectral flatness (near 0)
    assert fv.spectral_flatness < 0.3


def test_extract_spectral_flatness_high_for_noise(extractor):
    rng = np.random.default_rng(42)
    spectrum = rng.normal(-60, 1, 1024)  # Flat noise
    fv = extractor.extract(spectrum, 400, 600)
    # Noise-like signal should have high spectral flatness (near 1)
    assert fv.spectral_flatness > 0.7


def test_extract_papr(extractor):
    spectrum = make_tone_spectrum()
    fv = extractor.extract(spectrum, 500, 524)
    assert fv.papr > 0  # Peak should exceed average


def test_extract_kurtosis(extractor):
    spectrum = make_tone_spectrum()
    fv = extractor.extract(spectrum, 500, 524)
    # Kurtosis is a float — just check it's computed
    assert isinstance(fv.kurtosis, float)


def test_extract_empty_region_raises(extractor):
    spectrum = make_tone_spectrum()
    with pytest.raises(ValueError):
        extractor.extract(spectrum, 500, 500)  # zero-width region
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_feature_extractor.py -v
```

Expected: FAIL — `feature_extractor` module not found.

- [ ] **Step 3: Implement FeatureExtractor**

```python
# backend/feature_extractor.py
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from models import FeatureVector


class FeatureExtractor:
    """Computes a 7-dimensional feature vector from a spectrum (dB) sub-region."""

    def extract(
        self,
        spectrum_db: np.ndarray,
        freq_start_bin: int,
        freq_end_bin: int,
    ) -> FeatureVector:
        """Extract features from a sub-region of the spectrum.

        Args:
            spectrum_db: Full spectrum array in dB (e.g., 1024 bins).
            freq_start_bin: Start bin index (inclusive).
            freq_end_bin: End bin index (exclusive).

        Returns:
            FeatureVector with 7 dimensions.
        """
        if freq_start_bin >= freq_end_bin:
            raise ValueError("freq_start_bin must be less than freq_end_bin")

        region = spectrum_db[freq_start_bin:freq_end_bin].astype(np.float64)
        n = len(region)

        # Convert to linear power for spectral computations
        linear = 10.0 ** (region / 10.0)

        # 1. Spectral centroid: weighted average position (normalized 0-1 within region)
        bins = np.arange(n, dtype=np.float64)
        total_power = np.sum(linear)
        if total_power > 0:
            centroid = np.sum(bins * linear) / total_power / max(n - 1, 1)
        else:
            centroid = 0.5

        # 2. Bandwidth at -3dB from peak
        peak_db = np.max(region)
        bandwidth_3db = float(np.sum(region >= peak_db - 3.0))

        # 3. Bandwidth at -10dB from peak
        bandwidth_10db = float(np.sum(region >= peak_db - 10.0))

        # 4. SNR: peak power vs mean power of the region
        mean_linear = np.mean(linear)
        peak_linear = np.max(linear)
        if mean_linear > 0:
            snr = 10.0 * np.log10(peak_linear / mean_linear)
        else:
            snr = 0.0

        # 5. PAPR: peak-to-average power ratio
        papr = snr  # Same computation for spectral data

        # 6. Spectral flatness: geometric mean / arithmetic mean of linear spectrum
        # Use log trick for numerical stability
        log_mean = np.mean(np.log(np.maximum(linear, 1e-20)))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(linear)
        if arith_mean > 0:
            spectral_flatness = float(geo_mean / arith_mean)
        else:
            spectral_flatness = 0.0
        spectral_flatness = max(0.0, min(1.0, spectral_flatness))

        # 7. Kurtosis of amplitude distribution
        kurt = float(scipy_kurtosis(region, fisher=True))

        return FeatureVector(
            values=[
                float(centroid),
                float(bandwidth_3db),
                float(bandwidth_10db),
                float(snr),
                float(papr),
                float(spectral_flatness),
                float(kurt),
            ]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_feature_extractor.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/feature_extractor.py backend/tests/test_feature_extractor.py
git commit -m "feat: add FeatureExtractor for 7-dim signal fingerprinting"
```

---

## Task 3: Emitter Database (SQLite)

**Files:**
- Create: `backend/emitter_db.py`
- Test: `backend/tests/test_emitter_db.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_emitter_db.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_emitter_db.py -v
```

Expected: FAIL — `emitter_db` module not found.

- [ ] **Step 3: Implement EmitterDB**

```python
# backend/emitter_db.py
import sqlite3
import json
from datetime import datetime, timedelta
from uuid import uuid4
from models import Emitter, Fingerprint, Observation, Baseline


class EmitterDB:
    """SQLite-backed storage for emitter library, fingerprints, observations, and baselines."""

    def __init__(self, db_path: str = "emitters.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS emitters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                freq_range_start REAL,
                freq_range_end REAL,
                notes TEXT DEFAULT '',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fingerprints (
                id TEXT PRIMARY KEY,
                emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
                feature_vector TEXT NOT NULL,
                quality_score REAL DEFAULT 1.0,
                captured_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                frequency REAL,
                power_db REAL,
                bandwidth REAL,
                is_active INTEGER DEFAULT 1,
                match_confidence REAL
            );

            CREATE TABLE IF NOT EXISTS baselines (
                emitter_id TEXT PRIMARY KEY REFERENCES emitters(id) ON DELETE CASCADE,
                power_mean REAL,
                power_std REAL,
                freq_mean REAL,
                freq_std REAL,
                typical_hours TEXT,
                duty_on_seconds REAL,
                duty_off_seconds REAL,
                updated_at TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- Emitter CRUD ---

    def create_emitter(
        self, name: str, tags: list[str], freq_start: float, freq_end: float
    ) -> Emitter:
        now = datetime.now().isoformat()
        emitter = Emitter(
            name=name,
            tags=tags,
            freq_range_start=freq_start,
            freq_range_end=freq_end,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            created_at=datetime.now(),
        )
        self.conn.execute(
            "INSERT INTO emitters (id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (emitter.id, emitter.name, json.dumps(emitter.tags),
             emitter.freq_range_start, emitter.freq_range_end, emitter.notes,
             now, now, now),
        )
        self.conn.commit()
        return emitter

    def get_emitter(self, emitter_id: str) -> Emitter | None:
        row = self.conn.execute(
            "SELECT id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at "
            "FROM emitters WHERE id = ?",
            (emitter_id,),
        ).fetchone()
        if row is None:
            return None
        return Emitter(
            id=row[0], name=row[1], tags=json.loads(row[2]),
            freq_range_start=row[3], freq_range_end=row[4], notes=row[5],
            first_seen=datetime.fromisoformat(row[6]),
            last_seen=datetime.fromisoformat(row[7]),
            created_at=datetime.fromisoformat(row[8]),
        )

    def list_emitters(self) -> list[Emitter]:
        rows = self.conn.execute(
            "SELECT id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at "
            "FROM emitters ORDER BY name"
        ).fetchall()
        return [
            Emitter(
                id=r[0], name=r[1], tags=json.loads(r[2]),
                freq_range_start=r[3], freq_range_end=r[4], notes=r[5],
                first_seen=datetime.fromisoformat(r[6]),
                last_seen=datetime.fromisoformat(r[7]),
                created_at=datetime.fromisoformat(r[8]),
            )
            for r in rows
        ]

    def update_emitter(
        self,
        emitter_id: str,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> Emitter:
        updates = []
        params: list = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if updates:
            params.append(emitter_id)
            self.conn.execute(
                f"UPDATE emitters SET {', '.join(updates)} WHERE id = ?", params
            )
            self.conn.commit()
        return self.get_emitter(emitter_id)

    def delete_emitter(self, emitter_id: str) -> None:
        self.conn.execute("DELETE FROM emitters WHERE id = ?", (emitter_id,))
        self.conn.commit()

    def update_last_seen(self, emitter_id: str) -> None:
        self.conn.execute(
            "UPDATE emitters SET last_seen = ? WHERE id = ?",
            (datetime.now().isoformat(), emitter_id),
        )
        self.conn.commit()

    # --- Fingerprints ---

    def add_fingerprint(
        self,
        emitter_id: str,
        feature_vector: list[float],
        quality_score: float = 1.0,
    ) -> Fingerprint:
        fp = Fingerprint(
            emitter_id=emitter_id,
            feature_vector=feature_vector,
            quality_score=quality_score,
        )
        self.conn.execute(
            "INSERT INTO fingerprints (id, emitter_id, feature_vector, quality_score, captured_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fp.id, fp.emitter_id, json.dumps(fp.feature_vector),
             fp.quality_score, fp.captured_at.isoformat()),
        )
        self.conn.commit()
        return fp

    def get_fingerprints(self, emitter_id: str) -> list[Fingerprint]:
        rows = self.conn.execute(
            "SELECT id, emitter_id, feature_vector, quality_score, captured_at "
            "FROM fingerprints WHERE emitter_id = ?",
            (emitter_id,),
        ).fetchall()
        return [
            Fingerprint(
                id=r[0], emitter_id=r[1], feature_vector=json.loads(r[2]),
                quality_score=r[3], captured_at=datetime.fromisoformat(r[4]),
            )
            for r in rows
        ]

    def get_all_fingerprints(self) -> list[Fingerprint]:
        rows = self.conn.execute(
            "SELECT id, emitter_id, feature_vector, quality_score, captured_at FROM fingerprints"
        ).fetchall()
        return [
            Fingerprint(
                id=r[0], emitter_id=r[1], feature_vector=json.loads(r[2]),
                quality_score=r[3], captured_at=datetime.fromisoformat(r[4]),
            )
            for r in rows
        ]

    # --- Observations ---

    def log_observation(
        self,
        emitter_id: str,
        frequency: float,
        power_db: float,
        bandwidth: float,
        is_active: bool = True,
        match_confidence: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        ts = (timestamp or datetime.now()).isoformat()
        self.conn.execute(
            "INSERT INTO observations (emitter_id, timestamp, frequency, power_db, bandwidth, is_active, match_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (emitter_id, ts, frequency, power_db, bandwidth, int(is_active), match_confidence),
        )
        self.conn.commit()

    def get_observations(
        self,
        emitter_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Observation]:
        query = "SELECT emitter_id, timestamp, frequency, power_db, bandwidth, is_active, match_confidence FROM observations WHERE emitter_id = ?"
        params: list = [emitter_id]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())
        query += " ORDER BY timestamp"
        rows = self.conn.execute(query, params).fetchall()
        return [
            Observation(
                emitter_id=r[0], timestamp=datetime.fromisoformat(r[1]),
                frequency=r[2], power_db=r[3], bandwidth=r[4],
                is_active=bool(r[5]), match_confidence=r[6],
            )
            for r in rows
        ]

    def prune_observations(self, max_age_hours: int = 24) -> int:
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        cursor = self.conn.execute(
            "DELETE FROM observations WHERE timestamp < ?", (cutoff,)
        )
        self.conn.commit()
        return cursor.rowcount

    # --- Baselines ---

    def compute_baseline(self, emitter_id: str, window_hours: int = 4) -> Baseline | None:
        since = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        rows = self.conn.execute(
            "SELECT frequency, power_db, timestamp, is_active FROM observations "
            "WHERE emitter_id = ? AND timestamp >= ? ORDER BY timestamp",
            (emitter_id, since),
        ).fetchall()

        if not rows:
            return None

        active_powers = [r[1] for r in rows if r[3]]
        active_freqs = [r[0] for r in rows if r[3]]

        import numpy as np

        power_mean = float(np.mean(active_powers)) if active_powers else 0.0
        power_std = float(np.std(active_powers)) if active_powers else 0.0
        freq_mean = float(np.mean(active_freqs)) if active_freqs else 0.0
        freq_std = float(np.std(active_freqs)) if active_freqs else 0.0

        # Typical hours: bucket by hour, find hours with >30% active rate
        from collections import Counter
        hour_counts = Counter()
        hour_active = Counter()
        for r in rows:
            h = datetime.fromisoformat(r[2]).hour
            hour_counts[h] += 1
            if r[3]:
                hour_active[h] += 1
        typical_hours = []
        for h in sorted(hour_counts):
            if hour_counts[h] > 0 and hour_active[h] / hour_counts[h] > 0.3:
                typical_hours.append([h, h + 1])

        baseline = Baseline(
            emitter_id=emitter_id,
            power_mean=power_mean,
            power_std=power_std,
            freq_mean=freq_mean,
            freq_std=freq_std,
            typical_hours=typical_hours,
            updated_at=datetime.now(),
        )

        # Upsert baseline
        self.conn.execute(
            "INSERT INTO baselines (emitter_id, power_mean, power_std, freq_mean, freq_std, typical_hours, duty_on_seconds, duty_off_seconds, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(emitter_id) DO UPDATE SET "
            "power_mean=excluded.power_mean, power_std=excluded.power_std, "
            "freq_mean=excluded.freq_mean, freq_std=excluded.freq_std, "
            "typical_hours=excluded.typical_hours, updated_at=excluded.updated_at",
            (emitter_id, power_mean, power_std, freq_mean, freq_std,
             json.dumps(typical_hours), 0.0, 0.0, baseline.updated_at.isoformat()),
        )
        self.conn.commit()
        return baseline

    def get_baseline(self, emitter_id: str) -> Baseline | None:
        row = self.conn.execute(
            "SELECT emitter_id, power_mean, power_std, freq_mean, freq_std, typical_hours, "
            "duty_on_seconds, duty_off_seconds, updated_at FROM baselines WHERE emitter_id = ?",
            (emitter_id,),
        ).fetchone()
        if row is None:
            return None
        return Baseline(
            emitter_id=row[0], power_mean=row[1], power_std=row[2],
            freq_mean=row[3], freq_std=row[4], typical_hours=json.loads(row[5]),
            duty_on_seconds=row[6], duty_off_seconds=row[7],
            updated_at=datetime.fromisoformat(row[8]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_emitter_db.py -v
```

Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/emitter_db.py backend/tests/test_emitter_db.py
git commit -m "feat: add EmitterDB with SQLite schema, CRUD, observations, baselines"
```

---

## Task 4: Emitter Matcher

**Files:**
- Create: `backend/emitter_matcher.py`
- Test: `backend/tests/test_emitter_matcher.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_emitter_matcher.py
import pytest
import os
import tempfile
import numpy as np
from emitter_db import EmitterDB
from emitter_matcher import EmitterMatcher
from models import FeatureVector


@pytest.fixture
def matcher():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    m = EmitterMatcher(db)
    yield m
    db.close()
    os.unlink(path)


def test_enroll_creates_emitter(matcher):
    vectors = [FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])]
    emitter = matcher.enroll("RADAR-A", ["hostile"], vectors, freq_start=99.0, freq_end=100.0)
    assert emitter.name == "RADAR-A"
    assert emitter.tags == ["hostile"]


def test_enroll_stores_fingerprint(matcher):
    vectors = [
        FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        FeatureVector(values=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]),
    ]
    emitter = matcher.enroll("RADAR-B", [], vectors, freq_start=99.0, freq_end=100.0)
    fps = matcher.db.get_fingerprints(emitter.id)
    assert len(fps) == 2


def test_match_identical_vector(matcher):
    vec = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    matcher.enroll("EXACT", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    results = matcher.match(vec)
    assert len(results) >= 1
    assert results[0].confidence > 0.99  # Identical vector = ~1.0 cosine sim
    assert results[0].emitter_name == "EXACT"


def test_match_similar_vector(matcher):
    vec = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    matcher.enroll("SIMILAR", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    query = FeatureVector(values=[1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05])
    results = matcher.match(query)
    assert len(results) >= 1
    assert results[0].confidence > 0.99  # Very similar


def test_match_orthogonal_vector(matcher):
    vec = FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    matcher.enroll("DIR-X", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    query = FeatureVector(values=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    assert len(results) >= 1
    assert results[0].confidence < 0.1  # Orthogonal = 0 cosine sim


def test_match_returns_sorted_by_confidence(matcher):
    matcher.enroll("A", [], [FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])], freq_start=99.0, freq_end=100.0)
    matcher.enroll("B", [], [FeatureVector(values=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])], freq_start=100.0, freq_end=101.0)
    matcher.load_library()
    query = FeatureVector(values=[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    # Results should be sorted descending by confidence
    for i in range(len(results) - 1):
        assert results[i].confidence >= results[i + 1].confidence


def test_match_empty_library(matcher):
    matcher.load_library()
    query = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    results = matcher.match(query)
    assert results == []


def test_enrich_adds_fingerprints(matcher):
    vec1 = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    emitter = matcher.enroll("ENRICH", [], [vec1], freq_start=99.0, freq_end=100.0)
    vec2 = FeatureVector(values=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])
    matcher.enrich(emitter.id, [vec2])
    fps = matcher.db.get_fingerprints(emitter.id)
    assert len(fps) == 2


def test_match_multiple_fingerprints_takes_best(matcher):
    vecs = [
        FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        FeatureVector(values=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    matcher.enroll("MULTI", [], vecs, freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    # Query close to second fingerprint
    query = FeatureVector(values=[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    assert len(results) == 1
    assert results[0].emitter_name == "MULTI"
    assert results[0].confidence > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_emitter_matcher.py -v
```

Expected: FAIL — `emitter_matcher` module not found.

- [ ] **Step 3: Implement EmitterMatcher**

```python
# backend/emitter_matcher.py
import numpy as np
from models import FeatureVector, Emitter, MatchResult
from emitter_db import EmitterDB


class EmitterMatcher:
    """Cosine-similarity matching engine for emitter fingerprints."""

    def __init__(self, db: EmitterDB):
        self.db = db
        # In-memory cache: list of (emitter_id, emitter_name, vector_np)
        self._library: list[tuple[str, str, np.ndarray]] = []

    def load_library(self) -> None:
        """Load all fingerprints into memory for fast matching."""
        fingerprints = self.db.get_all_fingerprints()
        emitters = {e.id: e for e in self.db.list_emitters()}
        self._library = []
        for fp in fingerprints:
            emitter = emitters.get(fp.emitter_id)
            if emitter:
                vec = np.array(fp.feature_vector, dtype=np.float64)
                self._library.append((emitter.id, emitter.name, vec))

    def match(self, vector: FeatureVector) -> list[MatchResult]:
        """Match a feature vector against the library.

        Returns matches sorted by confidence descending.
        For emitters with multiple fingerprints, takes the highest-scoring one.
        """
        if not self._library:
            return []

        query = vector.to_numpy()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query_unit = query / query_norm

        # Compute cosine similarity for each fingerprint
        # Group by emitter, keep best score
        best_scores: dict[str, tuple[str, float]] = {}  # emitter_id -> (name, score)
        for emitter_id, emitter_name, lib_vec in self._library:
            lib_norm = np.linalg.norm(lib_vec)
            if lib_norm == 0:
                continue
            sim = float(np.dot(query_unit, lib_vec / lib_norm))
            sim = max(0.0, sim)  # Clamp negative similarities
            if emitter_id not in best_scores or sim > best_scores[emitter_id][1]:
                best_scores[emitter_id] = (emitter_name, sim)

        results = [
            MatchResult(emitter_id=eid, emitter_name=name, confidence=round(score, 4))
            for eid, (name, score) in best_scores.items()
        ]
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def enroll(
        self,
        name: str,
        tags: list[str],
        vectors: list[FeatureVector],
        freq_start: float,
        freq_end: float,
    ) -> Emitter:
        """Create a new emitter and store its fingerprints."""
        emitter = self.db.create_emitter(name, tags, freq_start, freq_end)
        for vec in vectors:
            # Quality score based on SNR (dimension 3)
            quality = min(1.0, max(0.1, vec.snr / 40.0))
            self.db.add_fingerprint(emitter.id, vec.values, quality_score=quality)
        self.load_library()
        return emitter

    def enrich(self, emitter_id: str, vectors: list[FeatureVector]) -> None:
        """Add additional fingerprints to an existing emitter."""
        for vec in vectors:
            quality = min(1.0, max(0.1, vec.snr / 40.0))
            self.db.add_fingerprint(emitter_id, vec.values, quality_score=quality)
        self.load_library()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_emitter_matcher.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/emitter_matcher.py backend/tests/test_emitter_matcher.py
git commit -m "feat: add EmitterMatcher with cosine similarity matching and enrollment"
```

---

## Task 5: Anomaly Detector

**Files:**
- Create: `backend/anomaly_detector.py`
- Test: `backend/tests/test_anomaly_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_anomaly_detector.py
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
    # Create an emitter with a baseline
    emitter = db.create_emitter("TEST-E", [], 99.0, 100.0)
    # Add observations to create a baseline
    for i in range(20):
        db.log_observation(emitter.id, frequency=99.5, power_db=-50.0, bandwidth=20.0)
    db.compute_baseline(emitter.id)
    yield detector, db, emitter
    db.close()
    os.unlink(path)


def test_no_anomaly_within_baseline(setup):
    detector, db, emitter = setup
    obs = Observation(
        emitter_id=emitter.id,
        frequency=99.5,
        power_db=-50.0,
        bandwidth=20.0,
        match_confidence=0.95,
    )
    anomalies = detector.check(emitter.id, obs)
    assert len(anomalies) == 0


def test_power_anomaly(setup):
    detector, db, emitter = setup
    # Power way above baseline (-50 mean, std ~0)
    obs = Observation(
        emitter_id=emitter.id,
        frequency=99.5,
        power_db=-30.0,  # +20dB above baseline
        bandwidth=20.0,
        match_confidence=0.95,
    )
    anomalies = detector.check(emitter.id, obs)
    types = [a.type for a in anomalies]
    assert "power_anomaly" in types


def test_freq_shift_anomaly(setup):
    detector, db, emitter = setup
    obs = Observation(
        emitter_id=emitter.id,
        frequency=99.510,  # +10 kHz shift
        power_db=-50.0,
        bandwidth=20.0,
        match_confidence=0.95,
    )
    anomalies = detector.check(emitter.id, obs)
    types = [a.type for a in anomalies]
    assert "freq_shift" in types


def test_no_anomaly_small_freq_shift(setup):
    detector, db, emitter = setup
    obs = Observation(
        emitter_id=emitter.id,
        frequency=99.502,  # +2 kHz, below 5kHz threshold
        power_db=-50.0,
        bandwidth=20.0,
        match_confidence=0.95,
    )
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
    assert anomaly.severity == 0.3  # lower confidence = severity value
    db.close()
    os.unlink(path)


def test_no_new_emitter_anomaly_high_confidence():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    detector = AnomalyDetector(db)
    anomaly = detector.check_new_emitter(frequency=99.5, power_db=-40.0, confidence=0.9)
    assert anomaly is None  # High confidence = known emitter, no anomaly
    db.close()
    os.unlink(path)


def test_no_baseline_no_anomaly(setup):
    detector, db, emitter = setup
    # Create a new emitter with no baseline
    new_e = db.create_emitter("NEW", [], 100.0, 101.0)
    obs = Observation(
        emitter_id=new_e.id,
        frequency=100.5,
        power_db=-40.0,
        bandwidth=20.0,
    )
    anomalies = detector.check(new_e.id, obs)
    assert len(anomalies) == 0  # No baseline = can't detect anomalies
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_anomaly_detector.py -v
```

Expected: FAIL — `anomaly_detector` module not found.

- [ ] **Step 3: Implement AnomalyDetector**

```python
# backend/anomaly_detector.py
from models import Observation, Anomaly
from emitter_db import EmitterDB


class AnomalyDetector:
    """Checks observations against baselines to detect anomalous emitter behavior."""

    def __init__(
        self,
        db: EmitterDB,
        power_sigma_threshold: float = 2.0,
        freq_shift_threshold_khz: float = 5.0,
        new_emitter_confidence_threshold: float = 0.6,
    ):
        self.db = db
        self.power_sigma_threshold = power_sigma_threshold
        self.freq_shift_threshold_khz = freq_shift_threshold_khz
        self.new_emitter_confidence_threshold = new_emitter_confidence_threshold

    def check(self, emitter_id: str, current: Observation) -> list[Anomaly]:
        """Check an observation against the emitter's baseline.

        Returns a list of anomalies detected (may be empty).
        """
        baseline = self.db.get_baseline(emitter_id)
        if baseline is None:
            return []

        anomalies: list[Anomaly] = []
        emitter = self.db.get_emitter(emitter_id)
        emitter_name = emitter.name if emitter else None

        # Power anomaly: deviation > N sigma
        if baseline.power_std > 0:
            deviation = abs(current.power_db - baseline.power_mean)
            sigma_ratio = deviation / baseline.power_std
            if sigma_ratio > self.power_sigma_threshold:
                anomalies.append(Anomaly(
                    type="power_anomaly",
                    emitter_id=emitter_id,
                    emitter_name=emitter_name,
                    severity=round(sigma_ratio, 2),
                    baseline_value=baseline.power_mean,
                    current_value=current.power_db,
                    message=f"Power {current.power_db:.1f} dBm deviates {sigma_ratio:.1f}σ from baseline {baseline.power_mean:.1f} dBm",
                ))
        else:
            # Zero std: any difference is anomalous if > 1dB
            if abs(current.power_db - baseline.power_mean) > 1.0:
                anomalies.append(Anomaly(
                    type="power_anomaly",
                    emitter_id=emitter_id,
                    emitter_name=emitter_name,
                    severity=abs(current.power_db - baseline.power_mean),
                    baseline_value=baseline.power_mean,
                    current_value=current.power_db,
                    message=f"Power {current.power_db:.1f} dBm deviates from baseline {baseline.power_mean:.1f} dBm (zero variance baseline)",
                ))

        # Frequency shift: centroid moved > threshold
        freq_shift_mhz = abs(current.frequency - baseline.freq_mean)
        freq_shift_khz = freq_shift_mhz * 1000.0
        if freq_shift_khz > self.freq_shift_threshold_khz:
            anomalies.append(Anomaly(
                type="freq_shift",
                emitter_id=emitter_id,
                emitter_name=emitter_name,
                severity=round(freq_shift_khz, 2),
                baseline_value=baseline.freq_mean,
                current_value=current.frequency,
                message=f"Frequency shifted {freq_shift_khz:.1f} kHz from baseline {baseline.freq_mean:.3f} MHz",
            ))

        # Schedule anomaly: active outside typical hours
        if baseline.typical_hours:
            current_hour = current.timestamp.hour
            in_typical = any(
                start <= current_hour < end
                for start, end in baseline.typical_hours
            )
            if not in_typical and current.is_active:
                anomalies.append(Anomaly(
                    type="schedule_anomaly",
                    emitter_id=emitter_id,
                    emitter_name=emitter_name,
                    severity=1.0,
                    current_value=float(current_hour),
                    message=f"Active at hour {current_hour}, outside typical schedule",
                ))

        return anomalies

    def check_new_emitter(
        self,
        frequency: float,
        power_db: float,
        confidence: float,
    ) -> Anomaly | None:
        """Check if a signal with low match confidence should be flagged as new/unknown."""
        if confidence >= self.new_emitter_confidence_threshold:
            return None

        return Anomaly(
            type="new_emitter",
            emitter_id=None,
            emitter_name=None,
            severity=round(confidence, 4),
            current_value=power_db,
            message=f"Unknown emitter at {frequency:.3f} MHz, best match confidence {confidence:.0%}",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_anomaly_detector.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/anomaly_detector.py backend/tests/test_anomaly_detector.py
git commit -m "feat: add AnomalyDetector for power, frequency, schedule, and new-emitter anomalies"
```

---

## Task 6: Backend Integration — REST Endpoints & WebSocket

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/alert_engine.py`
- Test: `backend/tests/test_api_emitters.py`

- [ ] **Step 1: Write failing API tests**

```python
# backend/tests/test_api_emitters.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Import app fresh for each test to reset global state."""
    import importlib
    import main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_list_emitters_empty(client):
    res = client.get("/api/emitters")
    assert res.status_code == 200
    assert res.json() == []


def test_enroll_emitter(client):
    res = client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "RADAR-A",
        "tags": ["hostile"],
        "capture_frames": 2,  # Low count for testing
    })
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "RADAR-A"
    assert "id" in data


def test_list_emitters_after_enroll(client):
    client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "RADAR-A",
        "tags": [],
        "capture_frames": 2,
    })
    res = client.get("/api/emitters")
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_get_emitter_detail(client):
    enroll_res = client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "DETAIL",
        "tags": [],
        "capture_frames": 2,
    })
    eid = enroll_res.json()["id"]
    res = client.get(f"/api/emitters/{eid}")
    assert res.status_code == 200
    data = res.json()
    assert data["emitter"]["name"] == "DETAIL"
    assert "fingerprints" in data


def test_update_emitter(client):
    enroll_res = client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "OLD",
        "tags": [],
        "capture_frames": 2,
    })
    eid = enroll_res.json()["id"]
    res = client.put(f"/api/emitters/{eid}", json={
        "name": "NEW",
        "tags": ["updated"],
        "notes": "changed",
    })
    assert res.status_code == 200
    assert res.json()["name"] == "NEW"


def test_delete_emitter(client):
    enroll_res = client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "DEL",
        "tags": [],
        "capture_frames": 2,
    })
    eid = enroll_res.json()["id"]
    res = client.delete(f"/api/emitters/{eid}")
    assert res.status_code == 200
    # Verify deleted
    res = client.get("/api/emitters")
    assert len(res.json()) == 0


def test_get_emitter_history(client):
    enroll_res = client.post("/api/emitters/enroll", json={
        "freq_start": 99.2,
        "freq_end": 99.6,
        "name": "HIST",
        "tags": [],
        "capture_frames": 2,
    })
    eid = enroll_res.json()["id"]
    res = client.get(f"/api/emitters/{eid}/history")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_api_emitters.py -v
```

Expected: FAIL — `/api/emitters` endpoints not defined.

- [ ] **Step 3: Update alert_engine.py to accept anomaly alerts**

Add a method to `AlertEngine` at line 60 (end of file):

```python
    def format_anomaly_alert(self, anomaly) -> Alert:
        """Convert an Anomaly into an Alert for the alert history."""
        return Alert(
            subband_id=anomaly.emitter_id or "unknown",
            subband_name=f"[ANOMALY] {anomaly.emitter_name or 'Unknown'}",
            power_db=anomaly.current_value,
            threshold_db=anomaly.baseline_value or 0.0,
        )
```

- [ ] **Step 4: Update main.py with new imports and globals**

Replace the imports and globals section of `backend/main.py` (lines 1–22) with:

```python
import asyncio
import struct
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import SpectrumConfig, SubBand, Alert, MessageType, DisplayMode, Observation
from signal_source import SimulatedSource
from signal_processor import SignalProcessor
from alert_engine import AlertEngine
from emitter_db import EmitterDB
from emitter_matcher import EmitterMatcher
from anomaly_detector import AnomalyDetector
from feature_extractor import FeatureExtractor


# Global state
source = SimulatedSource()
processor = SignalProcessor()
alert_engine = AlertEngine()
config = SpectrumConfig()
subbands: dict[str, SubBand] = {}
alert_history: list[Alert] = []

MAX_ALERT_HISTORY = 200

# IC Signal Diagnostics
emitter_db = EmitterDB("emitters.db")
matcher = EmitterMatcher(emitter_db)
anomaly_detector = AnomalyDetector(emitter_db)
feature_extractor = FeatureExtractor()

# Load library at startup
matcher.load_library()

# Observation timing
_last_observation_time: float = 0.0
OBSERVATION_INTERVAL = 5.0  # seconds

# Baseline recomputation timing
_last_baseline_time: float = 0.0
BASELINE_INTERVAL = 60.0  # seconds

# Pruning timing
_last_prune_time: float = 0.0
PRUNE_INTERVAL = 60.0  # seconds
```

- [ ] **Step 5: Add new REST endpoints to main.py**

Add the following after the existing `/api/alerts` endpoint (after line 86):

```python
# --- Emitter API ---


class EnrollRequest(BaseModel):
    freq_start: float
    freq_end: float
    name: str
    tags: list[str] = []
    capture_frames: int = 32


class UpdateEmitterRequest(BaseModel):
    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


class EnrichRequest(BaseModel):
    freq_start: float
    freq_end: float
    capture_frames: int = 32


@app.post("/api/emitters/enroll")
async def enroll_emitter(req: EnrollRequest):
    """Capture N frames, extract features, enroll as new emitter."""
    vectors = []
    start_bin = processor.freq_to_bin(req.freq_start, config.center_freq, config.bandwidth)
    end_bin = processor.freq_to_bin(req.freq_end, config.center_freq, config.bandwidth)

    for _ in range(req.capture_frames):
        samples = source.get_samples(config.fft_size)
        spectrum = processor.compute_spectrum(samples)
        fv = feature_extractor.extract(spectrum, start_bin, end_bin)
        vectors.append(fv)
        await asyncio.sleep(1.0 / config.fps)

    emitter = matcher.enroll(req.name, req.tags, vectors, req.freq_start, req.freq_end)
    return emitter.model_dump(mode="json")


@app.get("/api/emitters")
async def list_emitters():
    return [e.model_dump(mode="json") for e in emitter_db.list_emitters()]


@app.get("/api/emitters/{emitter_id}")
async def get_emitter_detail(emitter_id: str):
    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Emitter not found")
    fingerprints = emitter_db.get_fingerprints(emitter_id)
    baseline = emitter_db.get_baseline(emitter_id)
    return {
        "emitter": emitter.model_dump(mode="json"),
        "fingerprints": [fp.model_dump(mode="json") for fp in fingerprints],
        "baseline": baseline.model_dump(mode="json") if baseline else None,
    }


@app.put("/api/emitters/{emitter_id}")
async def update_emitter(emitter_id: str, req: UpdateEmitterRequest):
    updated = emitter_db.update_emitter(emitter_id, name=req.name, tags=req.tags, notes=req.notes)
    if updated is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Emitter not found")
    return updated.model_dump(mode="json")


@app.delete("/api/emitters/{emitter_id}")
async def delete_emitter(emitter_id: str):
    emitter_db.delete_emitter(emitter_id)
    matcher.load_library()  # Refresh in-memory library
    return {"ok": True}


@app.get("/api/emitters/{emitter_id}/history")
async def get_emitter_history(
    emitter_id: str,
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
):
    from datetime import datetime as dt
    since_dt = dt.fromisoformat(since) if since else None
    until_dt = dt.fromisoformat(until) if until else None
    obs = emitter_db.get_observations(emitter_id, since=since_dt, until=until_dt)
    return [o.model_dump(mode="json") for o in obs]


@app.post("/api/emitters/{emitter_id}/enrich")
async def enrich_emitter(emitter_id: str, req: EnrichRequest):
    """Capture additional fingerprints for an existing emitter."""
    vectors = []
    start_bin = processor.freq_to_bin(req.freq_start, config.center_freq, config.bandwidth)
    end_bin = processor.freq_to_bin(req.freq_end, config.center_freq, config.bandwidth)

    for _ in range(req.capture_frames):
        samples = source.get_samples(config.fft_size)
        spectrum = processor.compute_spectrum(samples)
        fv = feature_extractor.extract(spectrum, start_bin, end_bin)
        vectors.append(fv)
        await asyncio.sleep(1.0 / config.fps)

    matcher.enrich(emitter_id, vectors)
    emitter = emitter_db.get_emitter(emitter_id)
    return emitter.model_dump(mode="json")
```

- [ ] **Step 6: Update the WebSocket loop to include matching and anomaly detection**

Replace the WebSocket function in `main.py` (the `spectrum_ws` function) with:

```python
@app.websocket("/ws/spectrum")
async def spectrum_ws(ws: WebSocket):
    await ws.accept()
    import time
    global _last_observation_time, _last_baseline_time, _last_prune_time

    try:
        while True:
            now = time.time()

            # Generate and process samples
            samples = source.get_samples(config.fft_size)
            spectrum = processor.compute_spectrum(
                samples,
                display_mode=config.display_mode,
                averaging_count=config.averaging_count,
            )

            # Check sub-band alerts (existing behavior)
            sb_powers: dict[str, float] = {}
            for sb in subbands.values():
                start_bin = processor.freq_to_bin(
                    sb.freq_start, config.center_freq, config.bandwidth
                )
                end_bin = processor.freq_to_bin(
                    sb.freq_end, config.center_freq, config.bandwidth
                )
                sb_powers[sb.id] = processor.compute_subband_power(
                    spectrum, start_bin, end_bin
                )

            new_alerts = alert_engine.check(list(subbands.values()), sb_powers)
            for alert in new_alerts:
                alert_history.append(alert)

            # --- Emitter matching ---
            match_results_msg: dict = {}
            noise_floor_threshold = -60.0  # dBFS

            for sb in subbands.values():
                power = sb_powers.get(sb.id, -120.0)
                if power < noise_floor_threshold:
                    continue

                start_bin = processor.freq_to_bin(
                    sb.freq_start, config.center_freq, config.bandwidth
                )
                end_bin = processor.freq_to_bin(
                    sb.freq_end, config.center_freq, config.bandwidth
                )
                try:
                    fv = feature_extractor.extract(spectrum, start_bin, end_bin)
                except ValueError:
                    continue

                matches = matcher.match(fv)
                top_matches = matches[:3]  # Top 3

                is_new = not top_matches or top_matches[0].confidence < 0.6
                match_results_msg[sb.id] = {
                    "matches": [
                        {"emitter_id": m.emitter_id, "name": m.emitter_name, "confidence": m.confidence}
                        for m in top_matches
                    ],
                    "is_new": is_new,
                }

                # Log observations (every OBSERVATION_INTERVAL)
                if now - _last_observation_time >= OBSERVATION_INTERVAL and top_matches:
                    best = top_matches[0]
                    emitter_db.log_observation(
                        best.emitter_id,
                        frequency=fv.spectral_centroid,
                        power_db=power,
                        bandwidth=fv.bandwidth_3db,
                        match_confidence=best.confidence,
                    )
                    emitter_db.update_last_seen(best.emitter_id)

                    # Anomaly detection
                    obs = Observation(
                        emitter_id=best.emitter_id,
                        frequency=fv.spectral_centroid,
                        power_db=power,
                        bandwidth=fv.bandwidth_3db,
                        match_confidence=best.confidence,
                    )
                    anomalies = anomaly_detector.check(best.emitter_id, obs)
                    for anomaly in anomalies:
                        alert_history.append(alert_engine.format_anomaly_alert(anomaly))

                # New emitter anomaly
                if is_new and top_matches:
                    anomaly = anomaly_detector.check_new_emitter(
                        frequency=fv.spectral_centroid,
                        power_db=power,
                        confidence=top_matches[0].confidence if top_matches else 0.0,
                    )
                    if anomaly:
                        alert_history.append(alert_engine.format_anomaly_alert(anomaly))

            if now - _last_observation_time >= OBSERVATION_INTERVAL:
                _last_observation_time = now

            # Baseline recomputation
            if now - _last_baseline_time >= BASELINE_INTERVAL:
                _last_baseline_time = now
                for emitter in emitter_db.list_emitters():
                    emitter_db.compute_baseline(emitter.id)

            # Pruning
            if now - _last_prune_time >= PRUNE_INTERVAL:
                _last_prune_time = now
                emitter_db.prune_observations()

            # Trim alert history
            if len(alert_history) > MAX_ALERT_HISTORY:
                del alert_history[: len(alert_history) - MAX_ALERT_HISTORY]

            # --- Send messages ---

            # Send spectrum frame (0x01)
            try:
                header = struct.pack("B", MessageType.SPECTRUM)
                await asyncio.wait_for(
                    ws.send_bytes(header + spectrum.tobytes()), timeout=0.05
                )
            except asyncio.TimeoutError:
                pass

            # Send alerts (0x02)
            for alert in new_alerts:
                try:
                    alert_msg = struct.pack("B", MessageType.ALERT) + json.dumps(
                        alert.model_dump(mode="json")
                    ).encode()
                    await asyncio.wait_for(ws.send_bytes(alert_msg), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            # Send match results (0x03)
            if match_results_msg:
                try:
                    match_payload = json.dumps({"subbands": match_results_msg}).encode()
                    match_msg = struct.pack("B", MessageType.MATCH) + match_payload
                    await asyncio.wait_for(ws.send_bytes(match_msg), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            # Maintain target FPS
            await asyncio.sleep(1.0 / config.fps)

    except WebSocketDisconnect:
        pass
    except Exception:
        await ws.close()
```

- [ ] **Step 7: Run API tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_api_emitters.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 8: Run all backend tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/main.py backend/alert_engine.py backend/tests/test_api_emitters.py
git commit -m "feat: integrate emitter matching, anomaly detection, and new REST endpoints into main app"
```

---

## Task 7: Frontend Types & Signal Stream Updates

**Files:**
- Create: `frontend/src/types/emitter.ts`
- Modify: `frontend/src/types/signal.ts`
- Modify: `frontend/src/hooks/useSignalStream.ts`

- [ ] **Step 1: Create emitter TypeScript types**

```typescript
// frontend/src/types/emitter.ts
export interface Emitter {
  id: string
  name: string
  tags: string[]
  freq_range_start: number
  freq_range_end: number
  notes: string
  first_seen: string
  last_seen: string
  created_at: string
}

export interface MatchResult {
  emitter_id: string
  name: string
  confidence: number
}

export interface SubBandMatch {
  matches: MatchResult[]
  is_new: boolean
}

export interface MatchData {
  subbands: Record<string, SubBandMatch>
}

export interface Observation {
  emitter_id: string
  timestamp: string
  frequency: number
  power_db: number
  bandwidth: number
  is_active: boolean
  match_confidence: number
}

export interface Anomaly {
  type: 'power_anomaly' | 'freq_shift' | 'schedule_anomaly' | 'new_emitter'
  emitter_id: string | null
  emitter_name: string | null
  severity: number
  baseline_value: number | null
  current_value: number
  message: string
  timestamp: string
}
```

- [ ] **Step 2: Add MESSAGE_TYPE.MATCH to signal.ts**

In `frontend/src/types/signal.ts`, update the `MESSAGE_TYPE` constant:

```typescript
export const MESSAGE_TYPE = {
  SPECTRUM: 0x01,
  ALERT: 0x02,
  MATCH: 0x03,
} as const
```

- [ ] **Step 3: Update useSignalStream to parse 0x03 messages**

Add match data parsing to `frontend/src/hooks/useSignalStream.ts`. Update the interface and state:

Add to imports:
```typescript
import type { MatchData } from '../types/emitter'
```

Update `SignalStreamState` interface to add:
```typescript
  matchData: MatchData | null
```

Update initial state to add:
```typescript
    matchData: null,
```

Add handling for `MESSAGE_TYPE.MATCH` in the `ws.onmessage` handler, after the `MESSAGE_TYPE.ALERT` block:

```typescript
        } else if (msgType === MESSAGE_TYPE.MATCH) {
          const jsonStr = new TextDecoder().decode(data.slice(1))
          const matchData: MatchData = JSON.parse(jsonStr)
          setState(s => ({ ...s, matchData }))
        }
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0, no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types/emitter.ts frontend/src/types/signal.ts frontend/src/hooks/useSignalStream.ts
git commit -m "feat: add emitter types and 0x03 match message parsing to frontend"
```

---

## Task 8: Frontend API Client & Emitters Hook

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/hooks/useEmitters.ts`

- [ ] **Step 1: Add emitter API calls to api.ts**

Append to `frontend/src/services/api.ts`:

```typescript
import type { Emitter, Observation } from '../types/emitter'

// Add to the api object:

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
```

- [ ] **Step 2: Create useEmitters hook**

```typescript
// frontend/src/hooks/useEmitters.ts
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
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/hooks/useEmitters.ts
git commit -m "feat: add emitter API client and useEmitters hook"
```

---

## Task 9: Enrollment Modal

**Files:**
- Create: `frontend/src/components/config/EnrollModal.tsx`

- [ ] **Step 1: Create EnrollModal component**

```tsx
// frontend/src/components/config/EnrollModal.tsx
import { useState } from 'react'
import { Radio, X, Loader2 } from 'lucide-react'

interface Props {
  freqStart: number
  freqEnd: number
  enrolling: boolean
  onEnroll: (name: string, tags: string[]) => void
  onCancel: () => void
}

export function EnrollModal({ freqStart, freqEnd, enrolling, onEnroll, onCancel }: Props) {
  const [name, setName] = useState('')
  const [tagsInput, setTagsInput] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    const tags = tagsInput
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)
    onEnroll(name.trim(), tags)
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-80">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
            <Radio className="w-4 h-4 text-cyan-400" />
            Enroll Emitter
          </h3>
          <button onClick={onCancel} className="text-gray-500 hover:text-gray-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="text-xs text-gray-400 mb-3">
          {freqStart.toFixed(3)} – {freqEnd.toFixed(3)} MHz
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="RADAR-A"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none"
              autoFocus
              disabled={enrolling}
            />
          </div>

          <div>
            <label className="text-xs text-gray-500 block mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={tagsInput}
              onChange={e => setTagsInput(e.target.value)}
              placeholder="hostile, radar"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none"
              disabled={enrolling}
            />
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 px-3 py-1.5 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700"
              disabled={enrolling}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-3 py-1.5 text-xs rounded bg-cyan-600 text-white hover:bg-cyan-500 disabled:opacity-50 flex items-center justify-center gap-1"
              disabled={!name.trim() || enrolling}
            >
              {enrolling && <Loader2 className="w-3 h-3 animate-spin" />}
              {enrolling ? 'Capturing...' : 'Enroll'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/config/EnrollModal.tsx
git commit -m "feat: add EnrollModal for emitter enrollment UI"
```

---

## Task 10: Emitter Library Panel

**Files:**
- Create: `frontend/src/components/emitters/EmitterLibrary.tsx`

- [ ] **Step 1: Create the EmitterLibrary component**

```tsx
// frontend/src/components/emitters/EmitterLibrary.tsx
import { useState } from 'react'
import { Search, Trash2, Radio } from 'lucide-react'
import type { Emitter, MatchData } from '../../types/emitter'

interface Props {
  emitters: Emitter[]
  matchData: MatchData | null
  onDelete: (id: string) => void
}

export function EmitterLibrary({ emitters, matchData, onDelete }: Props) {
  const [search, setSearch] = useState('')

  // Build a map of emitter_id -> best confidence across all subbands
  const confidenceMap: Record<string, number> = {}
  if (matchData) {
    for (const sb of Object.values(matchData.subbands)) {
      for (const m of sb.matches) {
        if (!confidenceMap[m.emitter_id] || m.confidence > confidenceMap[m.emitter_id]) {
          confidenceMap[m.emitter_id] = m.confidence
        }
      }
    }
  }

  const filtered = emitters.filter(
    e =>
      e.name.toLowerCase().includes(search.toLowerCase()) ||
      e.tags.some(t => t.toLowerCase().includes(search.toLowerCase()))
  )

  const getConfidenceColor = (confidence: number | undefined) => {
    if (confidence === undefined) return 'text-gray-600'
    if (confidence >= 0.85) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceBg = (confidence: number | undefined) => {
    if (confidence === undefined) return 'bg-gray-800'
    if (confidence >= 0.85) return 'bg-green-950/30 border-green-900/30'
    if (confidence >= 0.6) return 'bg-yellow-950/30 border-yellow-900/30'
    return 'bg-red-950/30 border-red-900/30'
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <Radio className="w-3 h-3" />
        Emitter Library
      </h3>

      <div className="relative mb-2">
        <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search emitters..."
          className="w-full bg-gray-800 border border-gray-700 rounded pl-7 pr-2 py-1 text-xs text-gray-300 focus:border-cyan-500 focus:outline-none"
        />
      </div>

      <div className="space-y-1 max-h-48 overflow-y-auto">
        {filtered.length === 0 && (
          <p className="text-xs text-gray-600 italic">No emitters enrolled</p>
        )}
        {filtered.map(emitter => {
          const confidence = confidenceMap[emitter.id]
          return (
            <div
              key={emitter.id}
              className={`flex items-center gap-2 text-xs border rounded px-2 py-1 ${getConfidenceBg(confidence)}`}
            >
              <span className={`font-bold text-lg leading-none ${getConfidenceColor(confidence)}`}>
                {confidence !== undefined ? '\u25CF' : '\u25CB'}
              </span>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-gray-200 truncate">{emitter.name}</div>
                {emitter.tags.length > 0 && (
                  <div className="text-[10px] text-gray-500 truncate">
                    {emitter.tags.join(', ')}
                  </div>
                )}
              </div>
              {confidence !== undefined && (
                <span className={`text-[10px] font-mono ${getConfidenceColor(confidence)}`}>
                  {(confidence * 100).toFixed(0)}%
                </span>
              )}
              <button
                onClick={() => onDelete(emitter.id)}
                className="text-gray-600 hover:text-red-400 shrink-0"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create the directory and verify TypeScript compiles**

```bash
mkdir -p frontend/src/components/emitters
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/emitters/EmitterLibrary.tsx
git commit -m "feat: add EmitterLibrary sidebar panel with live match confidence"
```

---

## Task 11: Match Overlay on Spectrum

**Files:**
- Create: `frontend/src/components/spectrum/MatchOverlay.tsx`

- [ ] **Step 1: Create the MatchOverlay renderer**

This is a pure function that draws match badges on the spectrum canvas context, called from `SpectrumCanvas`'s render effect.

```tsx
// frontend/src/components/spectrum/MatchOverlay.tsx
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

    // Badge background
    let bgColor: string
    let textColor: string
    if (best.confidence >= 0.85) {
      bgColor = 'rgba(34, 197, 94, 0.8)'  // green
      textColor = '#ffffff'
    } else if (best.confidence >= 0.6) {
      bgColor = 'rgba(234, 179, 8, 0.8)'   // yellow
      textColor = '#000000'
    } else {
      bgColor = 'rgba(239, 68, 68, 0.8)'   // red
      textColor = '#ffffff'
    }

    const label = `${best.name} ${(best.confidence * 100).toFixed(0)}%`
    ctx.font = '10px monospace'
    const metrics = ctx.measureText(label)
    const badgeW = metrics.width + 8
    const badgeH = 14

    // Draw badge
    ctx.fillStyle = bgColor
    ctx.beginPath()
    ctx.roundRect(x - badgeW / 2, y - badgeH / 2, badgeW, badgeH, 3)
    ctx.fill()

    // Draw text
    ctx.fillStyle = textColor
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, x, y)

    // "NEW" pulsing indicator for unknown signals
    if (sbMatch.is_new) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'
      ctx.font = 'bold 9px monospace'
      ctx.fillText('NEW', x, y + 14)
    }
  }

  // Reset text alignment
  ctx.textAlign = 'start'
  ctx.textBaseline = 'alphabetic'
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/spectrum/MatchOverlay.tsx
git commit -m "feat: add MatchOverlay for rendering emitter badges on spectrum canvas"
```

---

## Task 12: Emitter Timeline

**Files:**
- Create: `frontend/src/components/emitters/EmitterTimeline.tsx`

- [ ] **Step 1: Create EmitterTimeline component**

```tsx
// frontend/src/components/emitters/EmitterTimeline.tsx
import type { Emitter, MatchData } from '../../types/emitter'

interface Props {
  emitters: Emitter[]
  matchData: MatchData | null
}

export function EmitterTimeline({ emitters, matchData }: Props) {
  if (emitters.length === 0) return null

  // Build active state per emitter from current match data
  const activeEmitters = new Set<string>()
  if (matchData) {
    for (const sb of Object.values(matchData.subbands)) {
      for (const m of sb.matches) {
        if (m.confidence >= 0.6) {
          activeEmitters.add(m.emitter_id)
        }
      }
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-2">
      <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">
        Emitter Activity
      </h3>
      <div className="space-y-0.5">
        {emitters.map(emitter => {
          const active = activeEmitters.has(emitter.id)
          return (
            <div key={emitter.id} className="flex items-center gap-2 text-[10px]">
              <span className="text-gray-400 w-20 truncate font-mono">{emitter.name}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    active ? 'bg-cyan-500' : 'bg-gray-700'
                  }`}
                  style={{ width: active ? '100%' : '0%' }}
                />
              </div>
              <span className={`w-6 text-right ${active ? 'text-cyan-400' : 'text-gray-600'}`}>
                {active ? 'ON' : 'OFF'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/emitters/EmitterTimeline.tsx
git commit -m "feat: add EmitterTimeline showing live emitter on/off state"
```

---

## Task 13: Update AlertPanel for Anomaly Alerts

**Files:**
- Modify: `frontend/src/components/alerts/AlertPanel.tsx`

- [ ] **Step 1: Update AlertPanel to distinguish anomaly alerts**

Replace the entire `AlertPanel` component in `frontend/src/components/alerts/AlertPanel.tsx`:

```tsx
import { Bell, AlertTriangle } from 'lucide-react'
import type { Alert } from '../../types/subband'

interface Props {
  alerts: Alert[]
}

export function AlertPanel({ alerts }: Props) {
  const isAnomaly = (alert: Alert) => alert.subband_name.startsWith('[ANOMALY]')

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
        {[...alerts].reverse().map(alert => {
          const anomaly = isAnomaly(alert)
          return (
            <div
              key={alert.id}
              className={`flex items-center gap-2 text-xs border rounded px-2 py-1 ${
                anomaly
                  ? 'bg-amber-950/30 border-amber-900/30'
                  : 'bg-red-950/30 border-red-900/30'
              }`}
            >
              {anomaly ? (
                <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />
              ) : (
                <Bell className="w-3 h-3 text-red-400 shrink-0" />
              )}
              <span className={`font-medium ${anomaly ? 'text-amber-300' : 'text-red-300'}`}>
                {anomaly ? alert.subband_name.replace('[ANOMALY] ', '') : alert.subband_name}
              </span>
              <span className="text-gray-500">
                {alert.power_db.toFixed(1)} dB {anomaly ? '' : `> ${alert.threshold_db} dB`}
              </span>
              <span className="text-gray-600 ml-auto text-[10px]">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/alerts/AlertPanel.tsx
git commit -m "feat: distinguish anomaly alerts (amber) from threshold alerts (red) in AlertPanel"
```

---

## Task 14: Update SpectrumCanvas with Match Overlay & Enroll Drag

**Files:**
- Modify: `frontend/src/components/spectrum/SpectrumCanvas.tsx`

- [ ] **Step 1: Update SpectrumCanvas props and imports**

Add to imports:
```typescript
import { renderMatchOverlay } from './MatchOverlay'
import type { MatchData } from '../../types/emitter'
```

Update the `Props` interface to add:
```typescript
  matchData?: MatchData | null
  onEnrollDrag?: (freqStart: number, freqEnd: number) => void
```

- [ ] **Step 2: Add match overlay rendering to the render effect**

After the crosshair rendering section (after the "Axis labels" section, around line 237), add:

```typescript
    // Match overlay badges
    renderMatchOverlay({
      ctx,
      matchData: matchData ?? null,
      subbands,
      xAtFreq,
      paddingTop: PADDING.top,
    })
```

Add `matchData` to the effect's dependency array.

- [ ] **Step 3: Modify drag behavior to support both sub-band and enroll**

Update the `handleMouseUp` callback to call both `onSubBandDrag` and `onEnrollDrag`:

```typescript
  const handleMouseUp = useCallback(() => {
    if (dragRange) {
      const start = Math.min(dragRange.start, dragRange.end)
      const end = Math.max(dragRange.start, dragRange.end)
      if (end - start > 0.01) {
        // Both callbacks get the range — the parent decides which modal to show
        onSubBandDrag?.(start, end)
        onEnrollDrag?.(start, end)
      }
    }
    dragStartRef.current = null
    setDragRange(null)
  }, [dragRange, onSubBandDrag, onEnrollDrag])
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/spectrum/SpectrumCanvas.tsx
git commit -m "feat: integrate MatchOverlay and enroll drag into SpectrumCanvas"
```

---

## Task 15: Wire Everything into App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Update App.tsx with all new components and state**

Replace the entire `frontend/src/App.tsx`:

```tsx
import { useState, useCallback } from 'react'
import './App.css'
import { ControlBar } from './components/layout/ControlBar'
import { SpectrumCanvas } from './components/spectrum/SpectrumCanvas'
import { WaterfallCanvas } from './components/spectrum/WaterfallCanvas'
import { SubBandEditor } from './components/config/SubBandEditor'
import { AlertPanel } from './components/alerts/AlertPanel'
import { EmitterLibrary } from './components/emitters/EmitterLibrary'
import { EmitterTimeline } from './components/emitters/EmitterTimeline'
import { EnrollModal } from './components/config/EnrollModal'
import { useSignalStream } from './hooks/useSignalStream'
import { useSubBands } from './hooks/useSubBands'
import { useEmitters } from './hooks/useEmitters'
import { DEFAULT_CONFIG } from './types/signal'
import type { SpectrumConfig } from './types/signal'
import { api } from './services/api'

function App() {
  const [config, setConfig] = useState<SpectrumConfig>(DEFAULT_CONFIG)
  const [streaming, setStreaming] = useState(false)
  const [pendingRange, setPendingRange] = useState<{ start: number; end: number } | null>(null)
  const [enrollRange, setEnrollRange] = useState<{ start: number; end: number } | null>(null)
  const [showDragChoice, setShowDragChoice] = useState<{ start: number; end: number } | null>(null)

  const { currentSpectrum, waterfallBuffer, connected, alerts, frameCount, matchData } =
    useSignalStream(config.fft_size, streaming)
  const { subbands, addSubBand, deleteSubBand } = useSubBands()
  const { emitters, enrolling, enroll, deleteEmitter } = useEmitters()

  const handleConfigChange = useCallback(
    async (newConfig: SpectrumConfig) => {
      setConfig(newConfig)
      if (streaming) {
        try {
          await api.updateConfig(newConfig)
        } catch {
          // Backend unreachable
        }
      }
    },
    [streaming]
  )

  const handleToggleStream = useCallback(async () => {
    if (!streaming) {
      try {
        await api.updateConfig(config)
      } catch {
        // Backend may not be ready
      }
    }
    setStreaming(s => !s)
  }, [streaming, config])

  const handleSubBandDrag = useCallback((freqStart: number, freqEnd: number) => {
    setShowDragChoice({ start: freqStart, end: freqEnd })
  }, [])

  const handleDragChoiceSubBand = useCallback(() => {
    if (showDragChoice) {
      setPendingRange(showDragChoice)
      setShowDragChoice(null)
    }
  }, [showDragChoice])

  const handleDragChoiceEnroll = useCallback(() => {
    if (showDragChoice) {
      setEnrollRange(showDragChoice)
      setShowDragChoice(null)
    }
  }, [showDragChoice])

  const handleEnroll = useCallback(
    async (name: string, tags: string[]) => {
      if (!enrollRange) return
      await enroll({
        freq_start: enrollRange.start,
        freq_end: enrollRange.end,
        name,
        tags,
      })
      setEnrollRange(null)
    },
    [enrollRange, enroll]
  )

  return (
    <div className="min-h-screen flex flex-col bg-gray-950">
      <ControlBar
        config={config}
        streaming={streaming}
        connected={connected}
        onConfigChange={handleConfigChange}
        onToggleStream={handleToggleStream}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Main visualization area */}
        <div className="flex-1 flex flex-col p-3 gap-2 min-w-0">
          <SpectrumCanvas
            spectrum={currentSpectrum}
            config={config}
            subbands={subbands}
            frameCount={frameCount}
            matchData={matchData}
            onSubBandDrag={handleSubBandDrag}
          />
          <WaterfallCanvas
            waterfallBuffer={waterfallBuffer}
            frameCount={frameCount}
          />
          <EmitterTimeline emitters={emitters} matchData={matchData} />
          {!streaming && !currentSpectrum && (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
              Press Start to begin signal monitoring
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-72 shrink-0 border-l border-gray-800 p-3 space-y-3 overflow-y-auto">
          <SubBandEditor
            subbands={subbands}
            onAdd={addSubBand}
            onDelete={deleteSubBand}
            pendingRange={pendingRange}
            onClearPending={() => setPendingRange(null)}
          />
          <EmitterLibrary
            emitters={emitters}
            matchData={matchData}
            onDelete={deleteEmitter}
          />
          <AlertPanel alerts={alerts} />
        </div>
      </div>

      {/* Drag choice modal */}
      {showDragChoice && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-64">
            <h3 className="text-sm font-semibold text-gray-200 mb-1">
              {showDragChoice.start.toFixed(3)} – {showDragChoice.end.toFixed(3)} MHz
            </h3>
            <p className="text-xs text-gray-400 mb-3">What would you like to do?</p>
            <div className="flex gap-2">
              <button
                onClick={handleDragChoiceSubBand}
                className="flex-1 px-3 py-1.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-500"
              >
                Create Sub-Band
              </button>
              <button
                onClick={handleDragChoiceEnroll}
                className="flex-1 px-3 py-1.5 text-xs rounded bg-cyan-600 text-white hover:bg-cyan-500"
              >
                Enroll Emitter
              </button>
            </div>
            <button
              onClick={() => setShowDragChoice(null)}
              className="w-full mt-2 px-3 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Enrollment modal */}
      {enrollRange && (
        <EnrollModal
          freqStart={enrollRange.start}
          freqEnd={enrollRange.end}
          enrolling={enrolling}
          onEnroll={handleEnroll}
          onCancel={() => setEnrollRange(null)}
        />
      )}
    </div>
  )
}

export default App
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire emitter library, match overlay, enrollment, and timeline into main layout"
```

---

## Task 16: End-to-End Verification

- [ ] **Step 1: Run all backend tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Verify frontend compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Exit 0.

- [ ] **Step 3: Start both servers and verify app loads**

```bash
make dev
```

Open browser. Verify:
1. Spectrum and waterfall render as before
2. EmitterLibrary panel visible in sidebar
3. EmitterTimeline visible below waterfall
4. Drag-selecting on spectrum shows choice modal (Sub-Band vs Enroll)
5. Enrolling an emitter captures and saves fingerprints
6. Match confidence badges appear on enrolled signals
7. Anomaly alerts appear in amber in the alert panel

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete IC signal diagnostics — emitter fingerprinting, matching, anomaly detection"
```
