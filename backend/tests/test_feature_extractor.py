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
