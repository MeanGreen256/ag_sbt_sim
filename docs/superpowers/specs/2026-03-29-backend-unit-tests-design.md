# Backend Unit Tests for Untested Modules

**Date:** 2026-03-29
**Scope:** Unit tests for `signal_processor.py`, `signal_source.py`, and `alert_engine.py`

## Context

The backend has 5 existing test files covering models, feature extraction, emitter DB, emitter matching, and anomaly detection. Three modules with significant logic remain untested: the signal processor (FFT, display modes, frequency mapping), the simulated signal source (IQ generation, carriers, bursts), and the alert engine (threshold hysteresis and debounce state machine). Adding tests for these modules closes the gap in backend unit test coverage.

## Conventions

Follow existing test patterns:
- Plain pytest functions (no classes), `test_` prefix
- `numpy.testing` assertions for array comparisons
- Fixtures for shared setup (e.g., processor instances)
- Tests in `backend/tests/` alongside existing test files
- No new dependencies required

---

## Test File 1: `test_signal_processor.py`

**Module under test:** `backend/signal_processor.py` (SignalProcessor class)

### Tests

1. **test_compute_spectrum_output_shape** — Output is float32 ndarray of length fft_size
2. **test_compute_spectrum_pure_tone_peak** — A single-frequency complex exponential produces a peak at the correct FFT bin
3. **test_compute_spectrum_dc_centered** — DC component appears at bin fft_size/2 (fftshift verification)
4. **test_display_mode_average_converges** — AVERAGE mode: output stabilizes after `averaging_count` frames of identical input
5. **test_display_mode_max_hold_monotonic** — MAX_HOLD: output values never decrease across successive frames
6. **test_display_mode_min_hold_monotonic** — MIN_HOLD: output values never increase across successive frames
7. **test_reset_clears_accumulated_state** — After reset(), next AVERAGE/MAX_HOLD/MIN_HOLD frame equals RAW frame
8. **test_set_fft_size_updates_window** — set_fft_size changes fft_size, window length, and resets state
9. **test_compute_subband_power_flat_spectrum** — Known flat dB spectrum returns that same dB value
10. **test_compute_subband_power_invalid_range** — freq_start_bin >= freq_end_bin returns -120.0
11. **test_freq_to_bin_center** — Center frequency maps to bin fft_size/2
12. **test_freq_to_bin_edges_clamp** — Frequencies beyond bandwidth clamp to 0 and fft_size-1

---

## Test File 2: `test_signal_source.py`

**Module under test:** `backend/signal_source.py` (SimulatedSource class)

### Tests

1. **test_get_samples_returns_complex64** — Output dtype is complex64
2. **test_get_samples_correct_length** — Output length matches requested n
3. **test_carriers_produce_spectral_peaks** — FFT of output shows peaks near expected carrier offset frequencies
4. **test_noise_floor_present** — Samples have non-zero imaginary and real variance even without carriers
5. **test_burst_toggling** — Over enough samples, burst signals alternate between on and off states
6. **test_set_center_freq** — Updates center_freq attribute
7. **test_set_bandwidth_updates_sample_rate** — set_bandwidth updates both bandwidth and sample_rate
8. **test_deterministic_with_seed** — Two fresh SimulatedSource instances with same seed produce identical samples

---

## Test File 3: `test_alert_engine.py`

**Module under test:** `backend/alert_engine.py` (AlertEngine class)

### Tests

1. **test_no_alert_below_threshold** — Power below threshold produces no alerts
2. **test_alert_fires_above_threshold** — Power above threshold produces an alert with correct fields
3. **test_alert_debounce** — Second threshold crossing within debounce_seconds produces no alert
4. **test_alert_fires_after_debounce** — Alert fires again after debounce window expires (mock time.time)
5. **test_hysteresis_stays_armed** — Power slightly below threshold (but above threshold - hysteresis) does not disarm
6. **test_hysteresis_disarms** — Power below threshold - hysteresis disarms; next threshold crossing fires new alert
7. **test_multiple_subbands_independent** — Each subband tracks its own armed/debounce state independently
8. **test_missing_subband_power_skipped** — Subband not in power dict is silently skipped
9. **test_format_anomaly_alert_mapping** — Anomaly fields map to correct Alert fields
10. **test_format_anomaly_alert_none_fields** — None emitter_id/name produce "unknown" fallback

---

## Verification

Run all tests from the backend directory:
```bash
cd backend && .venv/bin/python -m pytest tests/ -v
```

All existing tests should continue to pass. New tests should pass on first run since they test existing, working code.
