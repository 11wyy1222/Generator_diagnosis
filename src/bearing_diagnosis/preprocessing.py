from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def load_waveform(path: str | Path) -> np.ndarray:
    """Load the first numeric column from a text/CSV/NPY waveform file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise ValueError(f"waveform file does not exist: {input_path}")
    if input_path.suffix.lower() == ".npy":
        values = np.load(input_path, allow_pickle=False)
    else:
        values = None
        try:
            # Confirmed source CSVs are headerless numeric tables.  loadtxt is
            # substantially faster than genfromtxt for the 131072-row files.
            candidate = np.loadtxt(input_path, delimiter=",", dtype=float, usecols=0)
            if candidate.size and np.isfinite(candidate).any():
                values = candidate
        except (ValueError, OSError):
            pass
        for delimiter in (",", None):
            if values is not None:
                break
            try:
                # The source CSVs contain multiple numeric channels.  The data
                # contract explicitly designates the first column as the model
                # waveform, so avoid parsing the other columns into memory.
                candidate = np.genfromtxt(
                    input_path, delimiter=delimiter, dtype=float, usecols=0
                )
                if candidate.size and np.isfinite(candidate).any():
                    values = candidate
                    break
            except (ValueError, OSError):
                pass
        if values is None:
            raise ValueError(f"cannot parse waveform as numeric data: {input_path}")
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1)


def validate_waveform(signal: np.ndarray, expected_length: int | None = None) -> list[str]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    reasons: list[str] = []
    if x.size < 4:
        reasons.append("too_short_for_fft")
        return reasons
    if expected_length is not None and x.size != expected_length:
        reasons.append("inconsistent_waveform_length")
    if not np.isfinite(x).all():
        reasons.append("non_finite_value")
        return reasons
    if float(np.ptp(x)) <= np.finfo(float).eps:
        reasons.append("constant_or_fill_value")
    if x.size >= 20:
        unique_ratio = np.unique(x).size / x.size
        if unique_ratio < 0.01:
            reasons.append("suspected_communication_fill")
        low, high = float(np.min(x)), float(np.max(x))
        saturated = np.mean((x == low) | (x == high))
        if saturated > 0.05:
            reasons.append("severe_clipping_or_saturation")
    return reasons


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(x.size, dtype=np.float64)
    multiplier[0] = 1.0
    if x.size % 2 == 0:
        multiplier[x.size // 2] = 1.0
        multiplier[1 : x.size // 2] = 2.0
    else:
        multiplier[1 : (x.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def linear_spectrum(signal: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 4 or sampling_rate_hz <= 0:
        raise ValueError("valid sampling rate and at least four values are required")
    windowed = (x - np.mean(x)) * np.hanning(x.size)
    amplitude = np.abs(np.fft.rfft(windowed)) * (2.0 / max(float(np.sum(np.hanning(x.size))), 1.0))
    if amplitude.size:
        amplitude[0] *= 0.5
    if x.size % 2 == 0 and amplitude.size > 1:
        amplitude[-1] *= 0.5
    frequency = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate_hz)
    return frequency, amplitude


def native_spectra(signal: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = np.asarray(signal, dtype=np.float64).reshape(-1) - float(np.mean(signal))
    frequency, ordinary = linear_spectrum(centered, sampling_rate_hz)
    envelope = analytic_envelope(centered)
    envelope -= float(np.mean(envelope))
    envelope_frequency, envelope_spectrum = linear_spectrum(envelope, sampling_rate_hz)
    if not np.array_equal(frequency, envelope_frequency):
        raise RuntimeError("ordinary and envelope spectra use inconsistent axes")
    cutoff = sampling_rate_hz / 2.56
    keep = frequency <= cutoff
    return frequency[keep], ordinary[keep], envelope_spectrum[keep]


@dataclass(frozen=True)
class FrequencyGrid:
    axis_hz: np.ndarray
    f_max_hz: float
    delta_f_hz: float

    @classmethod
    def fit(
        cls,
        sampling_parameters: Iterable[tuple[float, int]],
        business_f_max_hz: float,
        max_required_theoretical_hz: float | None = None,
    ) -> "FrequencyGrid":
        params = [(float(fs), int(length)) for fs, length in sampling_parameters]
        if not params or any(fs <= 0 or length < 4 for fs, length in params):
            raise ValueError("normal training sampling parameters are missing or invalid")
        f_max = min(business_f_max_hz, min(fs / 2.56 for fs, _ in params))
        delta = max(fs / length for fs, length in params)
        if max_required_theoretical_hz is not None and f_max < 1.2 * max_required_theoretical_hz:
            raise ValueError("frequency grid does not cover 1.2x the maximum required theoretical frequency")
        axis = np.arange(0.0, f_max + delta / 2.0, delta, dtype=np.float64)
        return cls(axis_hz=axis, f_max_hz=float(f_max), delta_f_hz=float(delta))

    def transform(self, frequency: np.ndarray, ordinary: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        if frequency[-1] + self.delta_f_hz / 2 < self.f_max_hz:
            raise ValueError("sample spectrum does not cover the frozen frequency grid")
        return np.stack(
            [
                np.interp(self.axis_hz, frequency, ordinary),
                np.interp(self.axis_hz, frequency, envelope),
            ],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class PreprocessState:
    amplitude_p995: float
    frequency_grid: FrequencyGrid
    rpm_min: float
    rpm_max: float

    def normalize_time(self, signal: np.ndarray) -> np.ndarray:
        if not math.isfinite(self.amplitude_p995) or self.amplitude_p995 <= 0:
            raise ValueError("invalid training amplitude P99.5")
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        return ((x - float(np.mean(x))) / self.amplitude_p995).astype(np.float32)

    def normalize_rpm(self, rpm: float) -> float:
        return float((rpm - self.rpm_min) / (self.rpm_max - self.rpm_min))


def fit_amplitude_p995(signals: Iterable[np.ndarray]) -> float:
    values = [np.abs(np.asarray(signal, dtype=np.float64).reshape(-1)) for signal in signals]
    if not values:
        raise ValueError("normal training signals are required")
    p995 = float(np.percentile(np.concatenate(values), 99.5))
    if not math.isfinite(p995) or p995 <= 0:
        raise ValueError("normal training signals have invalid amplitude scale")
    return p995
