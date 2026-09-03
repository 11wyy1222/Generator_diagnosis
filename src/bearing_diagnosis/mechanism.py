from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .schemas import COMPONENTS

# Per spectrum: 5 * (peak, local background, peak/background, relative offset),
# harmonic coverage/sum/decay, and left/right sideband amplitudes, ratio,
# relative spacing and symmetry.
FEATURES_PER_SPECTRUM = 28
FEATURE_COUNT = FEATURES_PER_SPECTRUM * 2 + 3


@dataclass(frozen=True)
class MechanismEvidence:
    features: np.ndarray
    valid_mask: np.ndarray
    component_reliability: np.ndarray
    q_global: float
    theoretical_frequencies_hz: dict[str, list[float]]


def _local_evidence(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    shaft_hz: float,
    base_hz: float,
    fft_resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[float] = []
    mask: list[float] = []
    peaks: list[float] = []
    valid_harmonics = 0
    eps = np.finfo(float).eps
    for harmonic in range(1, 6):
        target = harmonic * base_hz
        half_width = max(3.0 * fft_resolution, 0.05 * target)
        valid = target > 0 and target + half_width <= frequency[-1]
        if not valid:
            features.extend((0.0, 0.0, 0.0, 0.0))
            mask.extend((0.0, 0.0, 0.0, 0.0))
            continue
        search = (frequency >= target - half_width) & (frequency <= target + half_width)
        indices = np.flatnonzero(search)
        peak_index = int(indices[np.argmax(spectrum[indices])])
        peak = float(spectrum[peak_index])
        local = (frequency >= max(0.0, target - 3 * half_width)) & (frequency <= target + 3 * half_width)
        exclude = (frequency >= target - half_width) & (frequency <= target + half_width)
        background_values = spectrum[local & ~exclude]
        background = float(np.median(background_values)) if background_values.size else float(np.median(spectrum))
        relative_offset = abs(float(frequency[peak_index]) - target) / max(target, eps)
        features.extend((peak, background, peak / max(background, eps), relative_offset))
        mask.extend((1.0, 1.0, 1.0, 1.0))
        peaks.append(peak)
        valid_harmonics += 1
    coverage = valid_harmonics / 5.0
    peak_sum = float(np.sum(peaks)) if peaks else 0.0
    decay = float(peaks[-1] / max(peaks[0], eps)) if len(peaks) >= 2 else 0.0
    base_valid = base_hz > shaft_hz and base_hz + shaft_hz <= frequency[-1]
    if base_valid:
        width = max(3.0 * fft_resolution, 0.05 * base_hz)
        def sample_peak(target: float) -> tuple[float, float]:
            indices = np.flatnonzero(np.abs(frequency - target) <= width)
            index = int(indices[np.argmax(spectrum[indices])])
            return float(spectrum[index]), float(frequency[index])

        (main, main_hz), (left, left_hz), (right, right_hz) = (
            sample_peak(base_hz), sample_peak(base_hz - shaft_hz), sample_peak(base_hz + shaft_hz)
        )
        sideband_ratio = (left + right) / max(2.0 * main, eps)
        sideband_spacing = ((main_hz - left_hz) + (right_hz - main_hz)) / max(2.0 * shaft_hz, eps)
        symmetry = 1.0 - abs(left - right) / max(left + right, eps)
    else:
        left = right = sideband_ratio = sideband_spacing = symmetry = 0.0
    has_peak = bool(peaks)
    features.extend((coverage, peak_sum, decay, left, right, sideband_ratio, sideband_spacing, symmetry))
    mask.extend(
        (
            float(has_peak), float(has_peak), float(len(peaks) >= 2),
            float(base_valid), float(base_valid), float(base_valid),
            float(base_valid), float(base_valid),
        )
    )
    return np.asarray(features, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def extract_mechanism_features(
    frequency_hz: np.ndarray,
    ordinary_spectrum: np.ndarray,
    envelope_spectrum: np.ndarray,
    rpm: float,
    component_orders: dict[str, float],
    sampling_rate_hz: float,
    waveform_length: int,
) -> MechanismEvidence:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    ordinary = np.asarray(ordinary_spectrum, dtype=np.float64)
    envelope = np.asarray(envelope_spectrum, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size < 2 or ordinary.shape != frequency.shape or envelope.shape != frequency.shape:
        raise ValueError("native spectrum arrays must be aligned one-dimensional arrays")
    shaft_hz = float(rpm) / 60.0
    fft_resolution = float(sampling_rate_hz) / int(waveform_length)
    all_features: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    reliability: list[float] = []
    theoretical: dict[str, list[float]] = {}
    for component in COMPONENTS:
        order = float(component_orders.get(component, 0.0))
        input_valid = math.isfinite(order) and order > 0 and math.isfinite(shaft_hz) and shaft_hz > 0
        base_hz = shaft_hz * order if input_valid else 0.0
        theoretical[component] = [base_hz * harmonic for harmonic in range(1, 6)]
        ordinary_features, ordinary_mask = _local_evidence(
            frequency, ordinary, shaft_hz, base_hz, fft_resolution
        )
        envelope_features, envelope_mask = _local_evidence(
            frequency, envelope, shaft_hz, base_hz, fft_resolution
        )
        coverage = sum(target <= frequency[-1] for target in theoretical[component]) / 5.0
        resolution_quality = min(1.0, base_hz / max(6.0 * fft_resolution, np.finfo(float).eps)) if input_valid else 0.0
        q_component = float(input_valid) * coverage * resolution_quality
        ordinary_ratio = float(np.mean(ordinary_features[2:20:4])) if ordinary_mask[2:20:4].any() else 0.0
        envelope_ratio = float(np.mean(envelope_features[2:20:4])) if envelope_mask[2:20:4].any() else 0.0
        ratio_valid = bool(ordinary_mask[2:20:4].any() and envelope_mask[2:20:4].any())
        spectral_consistency = (
            min(ordinary_ratio, envelope_ratio) / max(ordinary_ratio, envelope_ratio, np.finfo(float).eps)
            if ratio_valid else 0.0
        )
        combined = np.concatenate(
            [
                ordinary_features,
                envelope_features,
                np.asarray(
                    [
                        ordinary_ratio,
                        envelope_ratio,
                        spectral_consistency,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        combined_mask = np.concatenate(
            [ordinary_mask, envelope_mask, np.asarray([ordinary_mask.any(), envelope_mask.any(), ratio_valid], dtype=np.float32)]
        )
        all_features.append(combined)
        all_masks.append(combined_mask)
        reliability.append(q_component)
    q_values = np.asarray(reliability, dtype=np.float32)
    return MechanismEvidence(
        features=np.stack(all_features),
        valid_mask=np.stack(all_masks),
        component_reliability=q_values,
        q_global=float(np.mean(q_values)),
        theoretical_frequencies_hz=theoretical,
    )
