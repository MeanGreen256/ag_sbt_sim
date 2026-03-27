import asyncio
import struct
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from models import SpectrumConfig, SubBand, Alert, MessageType, DisplayMode
from signal_source import SimulatedSource
from signal_processor import SignalProcessor
from alert_engine import AlertEngine


# Global state
source = SimulatedSource()
processor = SignalProcessor()
alert_engine = AlertEngine()
config = SpectrumConfig()
subbands: dict[str, SubBand] = {}
alert_history: list[Alert] = []

MAX_ALERT_HISTORY = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="RF Sub-Band Monitor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST API ---


@app.get("/api/config")
async def get_config():
    return config


@app.post("/api/config")
async def update_config(new_config: SpectrumConfig):
    global config
    config = new_config
    source.set_center_freq(config.center_freq)
    source.set_bandwidth(config.bandwidth)
    processor.set_fft_size(config.fft_size)
    return config


@app.get("/api/subbands")
async def list_subbands():
    return list(subbands.values())


@app.post("/api/subbands")
async def create_subband(sb: SubBand):
    subbands[sb.id] = sb
    return sb


@app.put("/api/subbands/{sb_id}")
async def update_subband(sb_id: str, sb: SubBand):
    sb.id = sb_id
    subbands[sb_id] = sb
    return sb


@app.delete("/api/subbands/{sb_id}")
async def delete_subband(sb_id: str):
    subbands.pop(sb_id, None)
    return {"ok": True}


@app.get("/api/alerts")
async def list_alerts():
    return alert_history[-50:]


# --- WebSocket ---


@app.websocket("/ws/spectrum")
async def spectrum_ws(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            # Generate and process samples
            samples = source.get_samples(config.fft_size)
            spectrum = processor.compute_spectrum(
                samples,
                display_mode=config.display_mode,
                averaging_count=config.averaging_count,
            )

            # Check sub-band alerts
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
            # Trim history
            if len(alert_history) > MAX_ALERT_HISTORY:
                del alert_history[: len(alert_history) - MAX_ALERT_HISTORY]

            # Backpressure: skip frame if send buffer is backed up
            try:
                # Send spectrum frame: 1 byte type + float32 array
                header = struct.pack("B", MessageType.SPECTRUM)
                await asyncio.wait_for(
                    ws.send_bytes(header + spectrum.tobytes()), timeout=0.05
                )
            except asyncio.TimeoutError:
                # Client can't keep up, drop this frame
                pass

            # Send any new alerts as JSON
            for alert in new_alerts:
                try:
                    alert_msg = struct.pack("B", MessageType.ALERT) + json.dumps(
                        alert.model_dump(mode="json")
                    ).encode()
                    await asyncio.wait_for(ws.send_bytes(alert_msg), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            # Maintain target FPS
            await asyncio.sleep(1.0 / config.fps)

    except WebSocketDisconnect:
        pass
    except Exception:
        await ws.close()
