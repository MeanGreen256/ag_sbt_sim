import numpy as np
import numpy.testing as npt
from signal_processor import SignalProcessor
from models import DisplayMode


def make_tone(fft_size, bin_idx):
    """Create complex samples with a single tone at the given FFT bin."""
    # bin_idx after fftshift: DC is at fft_size/2
    # Before shift, bin 0 = DC. fftshift moves bin k to k + fft_size/2 (mod N).
    # We want peak at bin_idx after shift, so pre-shift bin = bin_idx - fft_size//2
    k = (bin_idx - fft_size // 2) % fft_size
    n = np.arange(fft_size)
    return np.exp(2j * np.pi * k * n / fft_size).astype(np.complex64)


def test_compute_spectrum_output_shape():
    sp = SignalProcessor(fft_size=512)
    samples = np.zeros(512, dtype=np.complex64)
    result = sp.compute_spectrum(samples)
    assert result.dtype == np.float32
    assert result.shape == (512,)


def test_compute_spectrum_pure_tone_peak():
    fft_size = 1024
    sp = SignalProcessor(fft_size=fft_size)
    target_bin = 600
    samples = make_tone(fft_size, target_bin)
    result = sp.compute_spectrum(samples)
    peak_bin = int(np.argmax(result))
    assert abs(peak_bin - target_bin) <= 1


def test_compute_spectrum_dc_centered():
    fft_size = 256
    sp = SignalProcessor(fft_size=fft_size)
    # DC signal = constant value
    samples = np.ones(fft_size, dtype=np.complex64)
    result = sp.compute_spectrum(samples)
    peak_bin = int(np.argmax(result))
    assert peak_bin == fft_size // 2


def test_display_mode_average_converges():
    sp = SignalProcessor(fft_size=256)
    samples = np.random.default_rng(0).standard_normal(256).astype(np.complex64)
    avg_count = 10
    for _ in range(avg_count * 3):
        result = sp.compute_spectrum(samples, DisplayMode.AVERAGE, avg_count)
    # Feed one more identical frame — should barely change
    prev = result.copy()
    result2 = sp.compute_spectrum(samples, DisplayMode.AVERAGE, avg_count)
    npt.assert_allclose(result2, prev, atol=0.5)


def test_display_mode_max_hold_monotonic():
    sp = SignalProcessor(fft_size=256)
    rng = np.random.default_rng(1)
    prev = None
    for _ in range(10):
        samples = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
            np.complex64
        )
        result = sp.compute_spectrum(samples, DisplayMode.MAX_HOLD)
        if prev is not None:
            assert np.all(result >= prev - 1e-6)
        prev = result.copy()


def test_display_mode_min_hold_monotonic():
    sp = SignalProcessor(fft_size=256)
    rng = np.random.default_rng(2)
    prev = None
    for _ in range(10):
        samples = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
            np.complex64
        )
        result = sp.compute_spectrum(samples, DisplayMode.MIN_HOLD)
        if prev is not None:
            assert np.all(result <= prev + 1e-6)
        prev = result.copy()


def test_reset_clears_accumulated_state():
    sp = SignalProcessor(fft_size=256)
    rng = np.random.default_rng(3)
    samples = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
        np.complex64
    )
    # Accumulate some MAX_HOLD state
    for _ in range(5):
        sp.compute_spectrum(samples, DisplayMode.MAX_HOLD)

    sp.reset()

    # After reset, MAX_HOLD on fresh input should equal RAW
    samples2 = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
        np.complex64
    )
    max_result = sp.compute_spectrum(samples2, DisplayMode.MAX_HOLD)

    sp.reset()
    raw_result = sp.compute_spectrum(samples2, DisplayMode.RAW)

    npt.assert_array_almost_equal(max_result, raw_result)


def test_set_fft_size_updates_window():
    sp = SignalProcessor(fft_size=256)
    assert sp.fft_size == 256

    sp.set_fft_size(512)
    assert sp.fft_size == 512
    assert len(sp._window) == 512
    # Accumulated state should be cleared
    assert sp._avg_buffer is None
    assert sp._max_hold is None
    assert sp._min_hold is None


def test_compute_subband_power_flat_spectrum():
    sp = SignalProcessor(fft_size=1024)
    # Flat spectrum at -30 dB
    flat_db = np.full(1024, -30.0, dtype=np.float32)
    power = sp.compute_subband_power(flat_db, 100, 200)
    assert abs(power - (-30.0)) < 0.01


def test_compute_subband_power_invalid_range():
    sp = SignalProcessor(fft_size=1024)
    spectrum = np.zeros(1024, dtype=np.float32)
    assert sp.compute_subband_power(spectrum, 200, 200) == -120.0
    assert sp.compute_subband_power(spectrum, 300, 100) == -120.0


def test_freq_to_bin_center():
    sp = SignalProcessor(fft_size=1024)
    center = 100.0
    bw = 2.4
    bin_idx = sp.freq_to_bin(center, center, bw)
    assert bin_idx == 512


def test_freq_to_bin_edges_clamp():
    sp = SignalProcessor(fft_size=1024)
    center = 100.0
    bw = 2.4
    # Way below range
    assert sp.freq_to_bin(50.0, center, bw) == 0
    # Way above range
    assert sp.freq_to_bin(150.0, center, bw) == 1023
