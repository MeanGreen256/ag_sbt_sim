# RF Sub-Band Signal Monitor

A web-based RF sub-band signal monitoring tool with real-time spectrum analysis, waterfall display, configurable sub-band monitoring, and threshold-based alerts.

![Architecture: Python + React](https://img.shields.io/badge/stack-Python%20%2B%20React-blue)

## Features

- **Real-time spectrum analyzer** — frequency vs power line chart with crosshair cursor readout
- **Scrolling waterfall display** — spectrogram with viridis colormap showing signal activity over time
- **Configurable sub-bands** — define frequency regions with visual overlays; drag-to-define on the spectrum
- **Alert engine** — threshold monitoring with hysteresis and debounce; live alert feed
- **Display modes** — raw, running average, max hold, min hold
- **Simulated RF source** — synthetic IQ signals with carriers, noise floor, and intermittent bursts
- **Hardware-ready architecture** — swap in RTL-SDR or HackRF with a single source class

## Architecture

```
Browser (React + Vite + Tailwind)
├── SpectrumCanvas        — freq vs power (Canvas 2D)
├── WaterfallCanvas       — scrolling spectrogram (Canvas 2D)
├── SubBandOverlay        — colored frequency regions
├── SubBandEditor         — CRUD for sub-band definitions
├── AlertPanel            — live alert feed
└── useSignalStream hook  — WebSocket + binary parsing + ring buffer
        │  WebSocket (binary Float32Array)
        ▼
Python Backend (FastAPI)
├── SignalSource (abstract) → SimulatedSource
├── SignalProcessor        — windowed FFT, PSD
├── AlertEngine            — threshold + hysteresis + debounce
└── REST API               — sub-bands, alerts, config
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install

```bash
make install
```

Or manually:

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Run

```bash
make dev
```

Or start each server separately:

```bash
# Terminal 1 — Backend (port 8000)
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (port 5173)
cd frontend
npm run dev
```

Open http://localhost:5173 and click **Start** to begin monitoring.

## Usage

1. **Start streaming** — click the Start button in the control bar
2. **Adjust parameters** — center frequency, bandwidth, FFT size, FPS, display mode
3. **Define sub-bands** — drag on the spectrum canvas or use the sidebar form
4. **Monitor alerts** — set thresholds per sub-band; alerts appear in the sidebar panel

## Tech Stack

**Backend:** FastAPI, NumPy, SciPy, WebSockets, Pydantic

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, Canvas 2D, Lucide icons

## Key Technical Decisions

- **Binary WebSocket** — Float32Array frames (4KB) instead of JSON (15KB) at 20 FPS
- **Canvas 2D** — sufficient for 1024 bins at 20 FPS, simpler than WebGL
- **IQ simulation** — generates complex baseband samples through full DSP pipeline
- **Backpressure handling** — server drops frames if client can't keep up
- **Vite proxy** — dev server proxies `/ws/*` and `/api/*` to backend

## Future: Hardware Integration

The `SignalSource` abstract class makes adding real SDR hardware straightforward:

```python
# backend/rtlsdr_source.py
from rtlsdr import RtlSdr
from signal_source import SignalSource

class RTLSDRSource(SignalSource):
    def __init__(self):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.gain = 'auto'

    def get_samples(self, n):
        return self.sdr.read_samples(n)
```

No changes needed to the signal processor, WebSocket streaming, or frontend.

## License

MIT
