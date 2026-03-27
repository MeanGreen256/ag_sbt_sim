from abc import ABC, abstractmethod
import numpy as np


class SignalSource(ABC):
    """Abstract base class for RF signal sources producing complex IQ samples."""

    @abstractmethod
    def get_samples(self, n: int) -> np.ndarray:
        """Return n complex IQ samples as a numpy array of complex64."""
        ...

    @abstractmethod
    def set_center_freq(self, freq_mhz: float) -> None:
        ...

    @abstractmethod
    def set_bandwidth(self, bw_mhz: float) -> None:
        ...


class SimulatedSource(SignalSource):
    """Generates synthetic RF signals for development and testing.

    Produces a configurable noise floor with continuous carrier signals
    and intermittent/bursty signals to simulate real RF activity.
    """

    def __init__(
        self,
        center_freq: float = 100.0,
        bandwidth: float = 2.4,
        sample_rate: float = 2.4e6,
    ):
        self.center_freq = center_freq  # MHz
        self.bandwidth = bandwidth  # MHz
        self.sample_rate = sample_rate  # Hz
        self._phase = 0.0
        self._burst_counter = 0
        self._burst_active = False
        self._rng = np.random.default_rng(42)

        # Continuous carriers: offset from center in MHz, amplitude
        self.carriers = [
            {"offset_mhz": -0.8, "amplitude": 0.5},
            {"offset_mhz": 0.3, "amplitude": 0.3},
            {"offset_mhz": 0.9, "amplitude": 0.7},
        ]

        # Intermittent signals
        self.bursts = [
            {
                "offset_mhz": -0.4,
                "amplitude": 0.6,
                "on_samples": int(sample_rate * 0.5),
                "off_samples": int(sample_rate * 1.5),
            },
            {
                "offset_mhz": 0.6,
                "amplitude": 0.4,
                "on_samples": int(sample_rate * 0.3),
                "off_samples": int(sample_rate * 2.0),
            },
        ]
        self._burst_phases = [0.0] * len(self.bursts)
        self._burst_counters = [0] * len(self.bursts)
        self._burst_states = [False] * len(self.bursts)

        self.noise_amplitude = 0.02

    def set_center_freq(self, freq_mhz: float) -> None:
        self.center_freq = freq_mhz

    def set_bandwidth(self, bw_mhz: float) -> None:
        self.bandwidth = bw_mhz
        self.sample_rate = bw_mhz * 1e6

    def get_samples(self, n: int) -> np.ndarray:
        t = np.arange(n) / self.sample_rate
        samples = np.zeros(n, dtype=np.complex64)

        # Add noise floor
        noise = self._rng.normal(0, self.noise_amplitude, n) + 1j * self._rng.normal(
            0, self.noise_amplitude, n
        )
        samples += noise.astype(np.complex64)

        # Add continuous carriers
        for carrier in self.carriers:
            freq_hz = carrier["offset_mhz"] * 1e6
            phase_inc = 2 * np.pi * freq_hz * t + self._phase
            sig = carrier["amplitude"] * np.exp(1j * phase_inc)
            samples += sig.astype(np.complex64)

        # Add intermittent burst signals
        for i, burst in enumerate(self.bursts):
            self._burst_counters[i] += n
            cycle = burst["on_samples"] + burst["off_samples"]
            pos_in_cycle = self._burst_counters[i] % cycle
            self._burst_states[i] = pos_in_cycle < burst["on_samples"]

            if self._burst_states[i]:
                freq_hz = burst["offset_mhz"] * 1e6
                phase_inc = 2 * np.pi * freq_hz * t + self._burst_phases[i]
                sig = burst["amplitude"] * np.exp(1j * phase_inc)
                samples += sig.astype(np.complex64)
                self._burst_phases[i] += 2 * np.pi * freq_hz * n / self.sample_rate

        self._phase += 2 * np.pi * n / self.sample_rate

        return samples
