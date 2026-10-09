"""Vibration-signal processing interface for MATLAB Python integration.

The input signal must already have been converted to engineering units
(for example, m/s^2).  Sensitivity conversion is intentionally handled by
the calling application rather than by this module.

MATLAB example
--------------
Add this file's folder to MATLAB's Python module search path, then call::

    module_path = "C:\\path\\to\\outputs";
    if count(py.sys.path, module_path) == 0
        insert(py.sys.path, int32(0), module_path);
    end
    result = py.vibration_analysis.vibration_analysis(signal, FS);

For MATLAB releases using the newer Python interface, convert the MATLAB
array before calling if required by the configured Python environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import hilbert


def vibration_analysis(signal: Any, fs: float) -> dict[str, np.ndarray]:
    """Process an engineering-unit vibration signal.

    Processing flow:
        1. Convert the input to a one-dimensional floating-point array.
        2. Remove the signal mean (DC component).
        3. Compute the original signal's Hann-windowed, single-sided RMS
           amplitude spectrum.
        4. Extract the Hilbert envelope directly from the DC-removed original
           time-domain signal; it does not use the original FFT result.
        5. Remove the envelope mean and compute its RMS amplitude spectrum.

    Parameters
    ----------
    signal
        One-dimensional signal already converted to physical units.
    fs
        Sampling frequency in Hz; must be positive.

    Returns
    -------
    dict
        ``time_signal``: DC-removed original time-domain signal.
        ``frequency``: Frequency axis of the original spectrum, in Hz.
        ``rms_spectrum``: Single-sided RMS amplitude spectrum of the original
            signal.
        ``envelope``: Hilbert envelope with its mean removed.
        ``envelope_frequency``: Frequency axis of the envelope spectrum, Hz.
        ``envelope_rms_spectrum``: Single-sided RMS amplitude spectrum of the
            DC-removed envelope.

    Notes
    -----
    The spectrum calculation matches the established MATLAB logic:
    symmetric Hann window -> divide by the window mean (coherent-gain
    correction) -> FFT -> single-sided amplitude correction -> divide by
    sqrt(2) to express non-DC lines as RMS amplitudes.  This is an amplitude
    spectrum intended for displaying discrete spectral components; use a PSD
    integration method for rigorous broadband band-RMS calculations.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be a finite positive sampling frequency in Hz.")

    raw_signal = _as_1d_signal(signal)
    time_signal = raw_signal - np.mean(raw_signal)

    frequency, rms_spectrum = _calculate_rms_spectrum(time_signal, fs)

    # This branch starts from the DC-removed original time signal, not FFT data.
    envelope = _calculate_envelope(time_signal)
    envelope_frequency, envelope_rms_spectrum = _calculate_rms_spectrum(
        envelope, fs
    )

    return {
        "time_signal": time_signal,
        "frequency": frequency,
        "rms_spectrum": rms_spectrum,
        "envelope": envelope,
        "envelope_frequency": envelope_frequency,
        "envelope_rms_spectrum": envelope_rms_spectrum,
    }


def _as_1d_signal(signal: Any) -> np.ndarray:
    """Convert supported Python/MATLAB array-like input to a finite 1-D array."""
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if array.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal must not contain NaN or infinite values.")
    return array


def _calculate_rms_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed, single-sided RMS amplitude spectrum."""
    length = signal.size
    window = np.hanning(length)  # Symmetric Hann window, matching MATLAB hann(L).
    coherent_gain = np.mean(window)

    if coherent_gain == 0.0:
        raise ValueError("Hann window coherent gain is zero.")

    windowed_signal = signal * window / coherent_gain
    fft_values = np.fft.rfft(windowed_signal)
    single_sided_peak = np.abs(fft_values) / length

    if length % 2 == 0:
        # DC and Nyquist bins do not have a negative-frequency counterpart.
        if single_sided_peak.size > 2:
            single_sided_peak[1:-1] *= 2.0
    else:
        # For an odd-length record, only the DC bin is excluded from doubling.
        single_sided_peak[1:] *= 2.0

    rms_spectrum = single_sided_peak / np.sqrt(2.0)
    frequency = np.fft.rfftfreq(length, d=1.0 / fs)
    return frequency, rms_spectrum


def _calculate_envelope(signal: np.ndarray) -> np.ndarray:
    """Extract the Hilbert envelope and remove its DC component."""
    envelope = np.abs(hilbert(signal))
    return envelope - np.mean(envelope)
