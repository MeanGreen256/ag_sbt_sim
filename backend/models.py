from pydantic import BaseModel, Field
from enum import Enum
from uuid import uuid4
from datetime import datetime


class DisplayMode(str, Enum):
    RAW = "raw"
    AVERAGE = "average"
    MAX_HOLD = "max_hold"
    MIN_HOLD = "min_hold"


class SpectrumConfig(BaseModel):
    center_freq: float = 100.0  # MHz
    bandwidth: float = 2.4  # MHz
    fft_size: int = 1024
    fps: int = 20
    display_mode: DisplayMode = DisplayMode.RAW
    averaging_count: int = 10


class SubBand(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    name: str
    freq_start: float  # MHz
    freq_end: float  # MHz
    color: str = "#3b82f6"
    threshold_db: float = -40.0


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    subband_id: str
    subband_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    power_db: float
    threshold_db: float


class MessageType:
    SPECTRUM = 0x01
    ALERT = 0x02
    MATCH = 0x03


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
