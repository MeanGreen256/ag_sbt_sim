import numpy as np
from scipy.signal import get_window
from models import DisplayMode


class SignalProcessor:
    """Computes power spectral density from IQ samples using windowed FFT."""

    def __init__(self, fft_size: int = 1024, window: str = "hann"):
        self.fft_size = fft_size
        self.window_name = window
        self._window = get_window(window, fft_size)
        self._window_correction = np.sum(self._window) ** 2

        # State for display modes
        self._avg_buffer: np.ndarray | None = None
        self._avg_count = 0
        self._max_hold: np.ndarray | None = None
        self._min_hold: np.ndarray | None = None

    def set_fft_size(self, size: int) -> None:
        self.fft_size = size
        self._window = get_window(self.window_name, size)
        self._window_correction = np.sum(self._window) ** 2
        self.reset()

    def reset(self) -> None:
        """Reset accumulated display mode state."""
        self._avg_buffer = None
        self._avg_count = 0
        self._max_hold = None
        self._min_hold = None

    def compute_spectrum(
        self,
        samples: np.ndarray,
        display_mode: DisplayMode = DisplayMode.RAW,
        averaging_count: int = 10,
    ) -> np.ndarray:
        """Compute power spectrum in dBFS from complex IQ samples.

        Returns an array of fft_size power values in dBFS, with DC centered
        (negative frequencies on left, positive on right).
        """
        # Take fft_size samples, apply window
        block = samples[: self.fft_size] * self._window

        # FFT and shift so DC is centered
        spectrum = np.fft.fftshift(np.fft.fft(block))

        # Power spectral density (magnitude squared), normalized
        psd = np.abs(spectrum) ** 2 / self._window_correction

        # Convert to dBFS, clamp minimum to avoid log(0)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))

        # Apply display mode
        match display_mode:
            case DisplayMode.RAW:
                return psd_db.astype(np.float32)

            case DisplayMode.AVERAGE:
                if self._avg_buffer is None:
                    self._avg_buffer = psd_db.copy()
                    self._avg_count = 1
                else:
                    self._avg_count = min(self._avg_count + 1, averaging_count)
                    alpha = 1.0 / self._avg_count
                    self._avg_buffer = (1 - alpha) * self._avg_buffer + alpha * psd_db
                return self._avg_buffer.astype(np.float32)

            case DisplayMode.MAX_HOLD:
                if self._max_hold is None:
                    self._max_hold = psd_db.copy()
                else:
                    self._max_hold = np.maximum(self._max_hold, psd_db)
                return self._max_hold.astype(np.float32)

            case DisplayMode.MIN_HOLD:
                if self._min_hold is None:
                    self._min_hold = psd_db.copy()
                else:
                    self._min_hold = np.minimum(self._min_hold, psd_db)
                return self._min_hold.astype(np.float32)

    def compute_subband_power(
        self,
        spectrum_db: np.ndarray,
        freq_start_bin: int,
        freq_end_bin: int,
    ) -> float:
        """Compute average power in a sub-band region (in dB)."""
        if freq_start_bin >= freq_end_bin:
            return -120.0
        subband = spectrum_db[freq_start_bin:freq_end_bin]
        # Average power in linear domain, then convert back to dB
        linear = 10.0 ** (subband / 10.0)
        return float(10.0 * np.log10(np.mean(linear)))

    def freq_to_bin(
        self, freq_mhz: float, center_freq: float, bandwidth: float
    ) -> int:
        """Convert a frequency in MHz to an FFT bin index."""
        offset = freq_mhz - center_freq
        normalized = (offset / bandwidth) + 0.5  # 0 to 1
        bin_idx = int(normalized * self.fft_size)
        return max(0, min(self.fft_size - 1, bin_idx))
