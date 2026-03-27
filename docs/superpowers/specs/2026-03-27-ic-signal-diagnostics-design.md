# IC Signal Diagnostics — Design Spec

## Overview

Extend the RF Sub-Band Signal Monitor with intelligence community signal diagnostic capabilities: emitter fingerprinting, live library matching, behavioral tracking, and anomaly detection. Built on feature extraction with cosine similarity matching and SQLite storage.

## Use Cases

1. **Passive collection & monitoring** — persistent surveillance of assigned spectrum
2. **Signal identification** — characterize unknown signals and match against a known emitter library
3. **Threat detection** — detect anomalous emitter behavior, new signals in denied bands, schedule deviations

## Design Decisions

- **Approach:** Feature extraction + SQLite fingerprint library with cosine similarity matching (not ML-based or rule-based)
- **Fingerprinting level:** Waveform/parameter-based first (bandwidth, SNR, spectral shape). Architecture supports adding hardware-level features (phase noise, I/Q imbalance) when higher-grade SDR hardware is added
- **Enrollment:** Live capture from the spectrum display + persistent library with real-time matching
- **Data retention:** Short-term (configurable, default 24h). SQLite with automatic pruning. No long-term archival in this version
- **Behavioral analysis:** Rolling baseline per emitter with threshold-based anomaly detection

---

## 1. Signal Feature Extraction Pipeline

### FeatureExtractor class (`backend/feature_extractor.py`)

Computes a 7-dimensional feature vector from IQ samples within a selected frequency region.

**Feature vector dimensions:**

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | Spectral centroid | Weighted average frequency of energy | MHz |
| 2 | Bandwidth (3dB) | Width of signal at -3dB from peak | kHz |
| 3 | Bandwidth (10dB) | Width of signal at -10dB from peak | kHz |
| 4 | SNR | Signal-to-noise ratio vs surrounding floor | dB |
| 5 | PAPR | Peak-to-average power ratio | dB |
| 6 | Spectral flatness | Geometric mean / arithmetic mean of spectrum (0=tone, 1=noise) | 0.0–1.0 |
| 7 | Kurtosis | Peakedness of amplitude distribution | float |

**Interface:**

```python
class FeatureExtractor:
    def extract(self, spectrum_db: np.ndarray, freq_start_bin: int, freq_end_bin: int) -> FeatureVector
    def extract_from_samples(self, iq_samples: np.ndarray, sample_rate: float) -> FeatureVector
```

`FeatureVector` is a Pydantic model wrapping a 7-element float array with named accessors for each dimension.

**Integration point:** Sits between the existing `SignalProcessor` (which produces the FFT) and the new `EmitterMatcher`. No changes to `SignalSource` or FFT computation.

**Extensibility:** Hardware-level features (phase noise, I/Q imbalance, turn-on transients) are added as dimensions 8+ to the vector. Matching logic uses cosine similarity which naturally handles variable-length vectors as long as both sides are padded/aligned.

---

## 2. Emitter Library & Matching

### SQLite Schema (`backend/emitter_db.py`)

Database file: `backend/emitters.db`

```sql
CREATE TABLE emitters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tags TEXT DEFAULT '[]',         -- JSON array: ["hostile", "radar", etc.]
    freq_range_start REAL,          -- MHz, typical observed range
    freq_range_end REAL,
    notes TEXT DEFAULT '',
    first_seen TEXT NOT NULL,       -- ISO timestamp
    last_seen TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE fingerprints (
    id TEXT PRIMARY KEY,
    emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
    feature_vector TEXT NOT NULL,   -- JSON float array [7 elements]
    quality_score REAL DEFAULT 1.0, -- 0.0–1.0, based on SNR at capture time
    captured_at TEXT NOT NULL
);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    frequency REAL,                 -- centroid MHz
    power_db REAL,
    bandwidth REAL,                 -- 3dB bandwidth kHz
    is_active INTEGER DEFAULT 1,
    match_confidence REAL
);

CREATE TABLE baselines (
    emitter_id TEXT PRIMARY KEY REFERENCES emitters(id) ON DELETE CASCADE,
    power_mean REAL,
    power_std REAL,
    freq_mean REAL,
    freq_std REAL,
    typical_hours TEXT,             -- JSON array of [start_hour, end_hour] pairs
    duty_on_seconds REAL,
    duty_off_seconds REAL,
    updated_at TEXT NOT NULL
);
```

**Retention:** `observations` table pruned on a configurable interval (default every 60s), deleting rows older than the retention window (default 24h).

### EmitterMatcher class (`backend/emitter_matcher.py`)

```python
class EmitterMatcher:
    def __init__(self, db: EmitterDB)
    def load_library(self) -> None              # load all fingerprints into memory
    def match(self, vector: FeatureVector) -> list[MatchResult]  # ranked by confidence
    def enroll(self, name: str, tags: list[str], vectors: list[FeatureVector]) -> Emitter
    def enrich(self, emitter_id: str, vectors: list[FeatureVector]) -> None
```

**Matching algorithm:**
1. Normalize both vectors to unit length
2. Cosine similarity against every fingerprint in the library
3. For emitters with multiple fingerprints, take the highest-scoring one
4. Return matches sorted by confidence descending

**Confidence thresholds:**
- **>85%** — positive match (green)
- **60–85%** — possible match (yellow)
- **<60%** — unknown / no match (red pulsing indicator)

**Performance:** Library loaded into memory at startup and refreshed on enrollment/deletion. Matching is O(N) where N = total fingerprints. For libraries under 1000 emitters, this runs sub-millisecond per sub-band.

**Match frequency:** Runs every frame, but only for sub-bands where the integrated power is above the noise floor (determined by a configurable threshold, default -60 dBFS). This prevents wasting cycles matching noise.

### Enrollment Flow

1. Analyst drag-selects a frequency region on the spectrum canvas
2. Existing behavior: creates a sub-band. New behavior: modal offers choice between "Create Sub-Band" and "Enroll Emitter"
3. If enrolling: system captures 32 consecutive frames (configurable), extracts a feature vector from each, averages them into a single high-quality fingerprint
4. Analyst enters name and tags in the enrollment modal
5. Quality score computed from the SNR during capture (higher SNR = more reliable fingerprint)
6. Emitter and fingerprint saved to SQLite, library refreshed in memory

---

## 3. Behavioral Tracking & Anomaly Detection

### Observation Logging

The main WebSocket loop records an observation for each matched emitter every 5 seconds (configurable `observation_interval`):

- Emitter ID, current timestamp
- Spectral centroid frequency, measured power, bandwidth
- Whether signal is currently active (power above noise floor)
- Match confidence from the most recent frame

### Baseline Computation (`backend/emitter_db.py`)

Baselines recomputed on a sliding window (default last 4h of observations). Recalculated every 60 seconds.

- **Power:** mean and standard deviation of `power_db` where `is_active = 1`
- **Frequency:** mean and standard deviation of `frequency` where `is_active = 1`
- **Typical hours:** cluster active timestamps into hour-of-day buckets, identify hours with >30% active rate
- **Duty cycle:** median on-duration and off-duration from contiguous active/inactive runs

### AnomalyDetector class (`backend/anomaly_detector.py`)

```python
class AnomalyDetector:
    def __init__(self, db: EmitterDB)
    def check(self, emitter_id: str, current: Observation) -> list[Anomaly]
```

**Anomaly types:**

| Type | Trigger | Severity |
|------|---------|----------|
| `power_anomaly` | Current power deviates >2σ from baseline mean | `abs(deviation) / σ` |
| `freq_shift` | Centroid moved >5kHz from baseline mean (configurable) | Shift magnitude in kHz |
| `schedule_anomaly` | Active outside typical hours, or absent during expected hours | Binary |
| `new_emitter` | Signal above noise floor with no library match (<60% confidence) | Match confidence (lower = more novel) |

**Anomaly model:**

```python
class Anomaly(BaseModel):
    type: str                # power_anomaly | freq_shift | schedule_anomaly | new_emitter
    emitter_id: str | None   # None for new_emitter
    emitter_name: str | None
    severity: float
    baseline_value: float | None
    current_value: float
    message: str             # human-readable description
    timestamp: datetime
```

Anomalies are routed through the existing `AlertEngine` as a new alert category. They appear in the same alert panel but with distinct styling (amber for behavioral anomalies vs red for threshold alerts).

---

## 4. API Endpoints

### New REST Endpoints

```
POST   /api/emitters/enroll           — { freq_start, freq_end, name, tags, capture_frames }
                                        Triggers N-frame capture, returns created Emitter

GET    /api/emitters                   — list all emitters [{id, name, tags, last_seen, freq_range}]
GET    /api/emitters/{id}              — full detail: emitter + fingerprints + baseline
PUT    /api/emitters/{id}              — update name, tags, notes
DELETE /api/emitters/{id}              — cascade delete: emitter, fingerprints, observations, baseline

GET    /api/emitters/{id}/history      — observation log, query params: since, until (ISO timestamps)
POST   /api/emitters/{id}/enrich       — { freq_start, freq_end, capture_frames }
                                        Capture additional fingerprints for existing emitter
```

### New WebSocket Message Type

**Type 0x03 — MATCH_RESULTS:**

```json
{
  "subbands": {
    "<subband_id>": {
      "matches": [
        {"emitter_id": "abc123", "name": "RADAR-A", "confidence": 0.94}
      ],
      "is_new": false
    }
  }
}
```

Sent as: 1-byte type prefix (0x03) + JSON payload. Sent every frame, but only includes sub-bands with signal above noise floor.

---

## 5. New Backend Files

| File | Class/Purpose |
|------|--------------|
| `backend/feature_extractor.py` | `FeatureExtractor` — computes 7-dim feature vectors from spectrum data |
| `backend/emitter_matcher.py` | `EmitterMatcher` — cosine similarity matching, enrollment, library management |
| `backend/emitter_db.py` | `EmitterDB` — SQLite schema, CRUD, observation logging, baseline computation, retention pruning |
| `backend/anomaly_detector.py` | `AnomalyDetector` — checks observations against baselines, generates anomaly alerts |

**Modified backend files:**

| File | Changes |
|------|---------|
| `backend/main.py` | New REST endpoints, integrate matcher + anomaly detector into WebSocket loop, add 0x03 message type |
| `backend/models.py` | Add `FeatureVector`, `Emitter`, `MatchResult`, `Observation`, `Anomaly` Pydantic models |
| `backend/alert_engine.py` | Accept anomaly alerts alongside threshold alerts |

---

## 6. New Frontend Components

| File | Purpose |
|------|---------|
| `frontend/src/components/config/EnrollModal.tsx` | Modal for naming/tagging emitter during enrollment |
| `frontend/src/components/emitters/EmitterLibrary.tsx` | Sidebar panel: searchable/filterable list of known emitters |
| `frontend/src/components/spectrum/MatchOverlay.tsx` | Emitter name + confidence badge on spectrum per sub-band |
| `frontend/src/components/emitters/EmitterTimeline.tsx` | Horizontal activity timeline below waterfall |
| `frontend/src/hooks/useEmitters.ts` | CRUD and enrollment API calls |
| `frontend/src/hooks/useMatchStream.ts` | Parses 0x03 WebSocket messages |
| `frontend/src/types/emitter.ts` | TypeScript interfaces: Emitter, MatchResult, Observation, Anomaly |

**Modified frontend files:**

| File | Changes |
|------|---------|
| `frontend/src/App.tsx` | Add EmitterLibrary to sidebar, EmitterTimeline below waterfall, wire up match data |
| `frontend/src/components/spectrum/SpectrumCanvas.tsx` | Integrate MatchOverlay rendering, modify drag behavior to offer enroll vs sub-band |
| `frontend/src/components/alerts/AlertPanel.tsx` | Distinguish anomaly alerts (amber) from threshold alerts (red) |
| `frontend/src/hooks/useSignalStream.ts` | Parse 0x03 message type alongside 0x01 and 0x02 |

---

## 7. UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Control Bar [Start/Stop] [Center] [BW] [FFT] [FPS] [Mode]         │
├───────────────────────────────────────────────┬─────────────────────┤
│                                               │ SUB-BANDS           │
│  Spectrum Canvas                              │  [existing panel]   │
│  + MatchOverlay (emitter badges on signals)   │                     │
│                                               ├─────────────────────┤
│                                               │ EMITTER LIBRARY     │
├───────────────────────────────────────────────┤  [search/filter]    │
│                                               │  ● RADAR-A  94%    │
│  Waterfall Canvas                             │  ● COMMS-7  61%    │
│                                               │  ○ UNKNOWN  new    │
│                                               ├─────────────────────┤
├───────────────────────────────────────────────┤ ALERTS              │
│  Emitter Timeline                             │  [threshold + anom] │
│  RADAR-A  ████░░████░░██                      │                     │
│  COMMS-7  ░░░░██░░░░██░░                      │                     │
└───────────────────────────────────────────────┴─────────────────────┘
```

---

## 8. Dependencies

**New backend dependencies:**
- None. SQLite is in Python stdlib. NumPy/SciPy (already installed) provide all needed math for feature extraction and cosine similarity.

**New frontend dependencies:**
- None. All new components use existing React + Tailwind + Canvas + Lucide stack.
