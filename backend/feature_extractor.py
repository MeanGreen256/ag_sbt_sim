import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from models import FeatureVector


class FeatureExtractor:
    """Computes a 7-dimensional feature vector from a spectrum (dB) sub-region."""

    def extract(
        self,
        spectrum_db: np.ndarray,
        freq_start_bin: int,
        freq_end_bin: int,
    ) -> FeatureVector:
        """Extract features from a sub-region of the spectrum.

        Args:
            spectrum_db: Full spectrum array in dB (e.g., 1024 bins).
            freq_start_bin: Start bin index (inclusive).
            freq_end_bin: End bin index (exclusive).

        Returns:
            FeatureVector with 7 dimensions.
        """
        if freq_start_bin >= freq_end_bin:
            raise ValueError("freq_start_bin must be less than freq_end_bin")

        region = spectrum_db[freq_start_bin:freq_end_bin].astype(np.float64)
        n = len(region)

        # Convert to linear power for spectral computations
        linear = 10.0 ** (region / 10.0)

        # 1. Spectral centroid: weighted average position (normalized 0-1 within region)
        bins = np.arange(n, dtype=np.float64)
        total_power = np.sum(linear)
        if total_power > 0:
            centroid = np.sum(bins * linear) / total_power / max(n - 1, 1)
        else:
            centroid = 0.5

        # 2. Bandwidth at -3dB from peak
        peak_db = np.max(region)
        bandwidth_3db = float(np.sum(region >= peak_db - 3.0))

        # 3. Bandwidth at -10dB from peak
        bandwidth_10db = float(np.sum(region >= peak_db - 10.0))

        # 4. SNR: peak power vs noise floor (median of linear power, robust to tones)
        peak_linear = np.max(linear)
        noise_floor_linear = np.median(linear)
        if noise_floor_linear > 0:
            snr = 10.0 * np.log10(peak_linear / noise_floor_linear)
        else:
            snr = 0.0

        # 5. PAPR: peak-to-average power ratio (peak vs arithmetic mean)
        mean_linear = np.mean(linear)
        if mean_linear > 0:
            papr = 10.0 * np.log10(peak_linear / mean_linear)
        else:
            papr = 0.0

        # 6. Spectral flatness: geometric mean / arithmetic mean of linear spectrum
        # Use log trick for numerical stability
        log_mean = np.mean(np.log(np.maximum(linear, 1e-20)))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(linear)
        if arith_mean > 0:
            spectral_flatness = float(geo_mean / arith_mean)
        else:
            spectral_flatness = 0.0
        spectral_flatness = max(0.0, min(1.0, spectral_flatness))

        # 7. Kurtosis of amplitude distribution
        kurt = float(scipy_kurtosis(region, fisher=True))

        return FeatureVector(
            values=[
                float(centroid),
                float(bandwidth_3db),
                float(bandwidth_10db),
                float(snr),
                float(papr),
                float(spectral_flatness),
                float(kurt),
            ]
        )
