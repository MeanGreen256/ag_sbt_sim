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
