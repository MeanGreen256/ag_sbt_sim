import asyncio
import struct
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
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

emitter_db = EmitterDB("emitters.db")
matcher = EmitterMatcher(emitter_db)
anomaly_detector = AnomalyDetector(emitter_db)
feature_extractor = FeatureExtractor()
matcher.load_library()

MAX_ALERT_HISTORY = 200

_last_observation_time: float = 0.0
OBSERVATION_INTERVAL = 5.0
_last_baseline_time: float = 0.0
BASELINE_INTERVAL = 60.0
_last_prune_time: float = 0.0
PRUNE_INTERVAL = 60.0


# --- Request models ---

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


# --- Emitter endpoints ---


@app.post("/api/emitters/enroll")
async def enroll_emitter(req: EnrollRequest):
    """Capture N frames, extract features, and enroll a new emitter."""
    start_bin = processor.freq_to_bin(req.freq_start, config.center_freq, config.bandwidth)
    end_bin = processor.freq_to_bin(req.freq_end, config.center_freq, config.bandwidth)
    if start_bin >= end_bin:
        raise HTTPException(status_code=400, detail="freq_start must be less than freq_end within the configured bandwidth")

    vectors = []
    for _ in range(req.capture_frames):
        samples = source.get_samples(config.fft_size)
        spectrum = processor.compute_spectrum(
            samples,
            display_mode=config.display_mode,
            averaging_count=config.averaging_count,
        )
        vec = feature_extractor.extract(spectrum, start_bin, end_bin)
        vectors.append(vec)

    emitter = matcher.enroll(req.name, req.tags, vectors, req.freq_start, req.freq_end)
    return emitter


@app.get("/api/emitters")
async def list_emitters():
    """List all enrolled emitters."""
    return emitter_db.list_emitters()


@app.get("/api/emitters/{emitter_id}")
async def get_emitter(emitter_id: str):
    """Get full emitter detail including fingerprints and baseline."""
    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail="Emitter not found")
    fingerprints = emitter_db.get_fingerprints(emitter_id)
    baseline = emitter_db.get_baseline(emitter_id)
    return {
        "emitter": emitter,
        "fingerprints": fingerprints,
        "baseline": baseline,
    }


@app.put("/api/emitters/{emitter_id}")
async def update_emitter(emitter_id: str, req: UpdateEmitterRequest):
    """Update emitter name, tags, or notes."""
    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail="Emitter not found")
    updated = emitter_db.update_emitter(emitter_id, name=req.name, tags=req.tags, notes=req.notes)
    return updated


@app.delete("/api/emitters/{emitter_id}")
async def delete_emitter(emitter_id: str):
    """Delete emitter (cascade) and refresh matcher library."""
    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail="Emitter not found")
    emitter_db.delete_emitter(emitter_id)
    matcher.load_library()
    return {"ok": True}


@app.get("/api/emitters/{emitter_id}/history")
async def get_emitter_history(
    emitter_id: str,
    since: Optional[str] = Query(default=None, description="ISO datetime lower bound"),
    until: Optional[str] = Query(default=None, description="ISO datetime upper bound"),
):
    """Get observation log for an emitter with optional time window."""
    from datetime import datetime

    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail="Emitter not found")

    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    observations = emitter_db.get_observations(emitter_id, since=since_dt, until=until_dt)
    return observations


@app.post("/api/emitters/{emitter_id}/enrich")
async def enrich_emitter(emitter_id: str, req: EnrichRequest):
    """Capture additional fingerprints for an existing emitter."""
    emitter = emitter_db.get_emitter(emitter_id)
    if emitter is None:
        raise HTTPException(status_code=404, detail="Emitter not found")

    start_bin = processor.freq_to_bin(req.freq_start, config.center_freq, config.bandwidth)
    end_bin = processor.freq_to_bin(req.freq_end, config.center_freq, config.bandwidth)
    if start_bin >= end_bin:
        raise HTTPException(status_code=400, detail="freq_start must be less than freq_end within the configured bandwidth")

    vectors = []
    for _ in range(req.capture_frames):
        samples = source.get_samples(config.fft_size)
        spectrum = processor.compute_spectrum(
            samples,
            display_mode=config.display_mode,
            averaging_count=config.averaging_count,
        )
        vec = feature_extractor.extract(spectrum, start_bin, end_bin)
        vectors.append(vec)

    matcher.enrich(emitter_id, vectors)
    return {"ok": True, "added_fingerprints": len(vectors)}


# --- WebSocket ---


@app.websocket("/ws/spectrum")
async def spectrum_ws(ws: WebSocket):
    global _last_observation_time, _last_baseline_time, _last_prune_time

    await ws.accept()

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

            # --- Emitter matching ---
            NOISE_FLOOR_DBM = -60.0
            match_results: dict[str, dict] = {}

            for sb in subbands.values():
                power = sb_powers.get(sb.id)
                if power is None or power < NOISE_FLOOR_DBM:
                    continue

                start_bin = processor.freq_to_bin(
                    sb.freq_start, config.center_freq, config.bandwidth
                )
                end_bin = processor.freq_to_bin(
                    sb.freq_end, config.center_freq, config.bandwidth
                )
                if start_bin >= end_bin:
                    continue

                try:
                    fv = feature_extractor.extract(spectrum, start_bin, end_bin)
                except Exception:
                    continue

                matches = matcher.match(fv)
                top_confidence = matches[0].confidence if matches else 0.0
                is_new = top_confidence < anomaly_detector.new_emitter_confidence_threshold

                match_results[sb.id] = {
                    "matches": [
                        {
                            "emitter_id": m.emitter_id,
                            "name": m.emitter_name,
                            "confidence": m.confidence,
                        }
                        for m in matches[:3]
                    ],
                    "is_new": is_new,
                }

                # Observation logging (throttled)
                if now - _last_observation_time >= OBSERVATION_INTERVAL:
                    if matches and not is_new:
                        top_match = matches[0]
                        freq_center = (sb.freq_start + sb.freq_end) / 2.0
                        obs = Observation(
                            emitter_id=top_match.emitter_id,
                            frequency=freq_center,
                            power_db=power,
                            bandwidth=(sb.freq_end - sb.freq_start) * 1000.0,
                            is_active=True,
                            match_confidence=top_match.confidence,
                        )
                        emitter_db.log_observation(
                            emitter_id=obs.emitter_id,
                            frequency=obs.frequency,
                            power_db=obs.power_db,
                            bandwidth=obs.bandwidth,
                            is_active=obs.is_active,
                            match_confidence=obs.match_confidence,
                        )
                        emitter_db.update_last_seen(top_match.emitter_id)

                        # Anomaly detection on observation
                        anomalies = anomaly_detector.check(top_match.emitter_id, obs)
                        for anomaly in anomalies:
                            anomaly_alert = alert_engine.format_anomaly_alert(anomaly)
                            alert_history.append(anomaly_alert)
                    elif is_new:
                        new_anomaly = anomaly_detector.check_new_emitter(
                            frequency=(sb.freq_start + sb.freq_end) / 2.0,
                            power_db=power,
                            confidence=top_confidence,
                        )
                        if new_anomaly:
                            anomaly_alert = alert_engine.format_anomaly_alert(new_anomaly)
                            alert_history.append(anomaly_alert)

            # Advance observation timer once per loop (not per subband)
            if now - _last_observation_time >= OBSERVATION_INTERVAL:
                _last_observation_time = now

            # Baseline recomputation (throttled)
            if now - _last_baseline_time >= BASELINE_INTERVAL:
                _last_baseline_time = now
                for emitter in emitter_db.list_emitters():
                    emitter_db.compute_baseline(emitter.id)

            # Observation pruning (throttled)
            if now - _last_prune_time >= PRUNE_INTERVAL:
                _last_prune_time = now
                emitter_db.prune_observations()

            # Trim alert history after anomaly alerts may have been appended
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

            # Send match results (0x03)
            if match_results:
                try:
                    match_msg = struct.pack("B", MessageType.MATCH) + json.dumps(
                        {"subbands": match_results}
                    ).encode()
                    await asyncio.wait_for(ws.send_bytes(match_msg), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            # Maintain target FPS
            await asyncio.sleep(1.0 / config.fps)

    except WebSocketDisconnect:
        pass
    except Exception:
        await ws.close()
