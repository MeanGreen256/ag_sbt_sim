import numpy as np
from signal_source import SimulatedSource


def test_get_samples_returns_complex64():
    src = SimulatedSource()
    samples = src.get_samples(1024)
    assert samples.dtype == np.complex64


def test_get_samples_correct_length():
    src = SimulatedSource()
    for n in [256, 1024, 4096]:
        assert len(src.get_samples(n)) == n


def test_carriers_produce_spectral_peaks():
    src = SimulatedSource(center_freq=100.0, bandwidth=2.4)
    samples = src.get_samples(4096)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1.0 / src.sample_rate))

    for carrier in src.carriers:
        expected_freq_hz = carrier["offset_mhz"] * 1e6
        # Find the bin closest to this carrier
        closest_bin = int(np.argmin(np.abs(freqs - expected_freq_hz)))
        # Check that a local peak exists near this bin (within ±5 bins)
        region = spectrum[max(0, closest_bin - 5) : closest_bin + 6]
        peak_in_region = np.max(region)
        median_spectrum = np.median(spectrum)
        assert peak_in_region > median_spectrum * 3, (
            f"No peak found near carrier at {carrier['offset_mhz']} MHz offset"
        )


def test_noise_floor_present():
    src = SimulatedSource()
    samples = src.get_samples(4096)
    # Even with carriers, there should be non-trivial variance from noise
    assert np.var(samples.real) > 0
    assert np.var(samples.imag) > 0


def test_burst_toggling():
    src = SimulatedSource()
    burst = src.bursts[0]
    cycle = burst["on_samples"] + burst["off_samples"]

    # Collect burst states over two full cycles
    states = []
    chunk = cycle // 10
    for _ in range(20):
        src.get_samples(chunk)
        states.append(src._burst_states[0])

    # Should have seen both True and False states
    assert True in states, "Burst never turned on"
    assert False in states, "Burst never turned off"


def test_set_center_freq():
    src = SimulatedSource(center_freq=100.0)
    src.set_center_freq(200.0)
    assert src.center_freq == 200.0


def test_set_bandwidth_updates_sample_rate():
    src = SimulatedSource(bandwidth=2.4)
    src.set_bandwidth(5.0)
    assert src.bandwidth == 5.0
    assert src.sample_rate == 5.0e6


def test_deterministic_with_seed():
    src1 = SimulatedSource()
    src2 = SimulatedSource()
    samples1 = src1.get_samples(1024)
    samples2 = src2.get_samples(1024)
    np.testing.assert_array_equal(samples1, samples2)
