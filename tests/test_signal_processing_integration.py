from __future__ import annotations

import json
from zipfile import ZipFile

import numpy as np
import pytest

from bearing_diagnosis.artifacts import load_preprocess_state, save_preprocess_state
from bearing_diagnosis.config import ModelConfig
from bearing_diagnosis.dataset import MechanismScaler
from bearing_diagnosis.preprocessing import (
    FrequencyGrid,
    PreprocessState,
    load_waveform,
    native_spectra,
    process_signal,
    process_time_signal,
)
from bearing_diagnosis.vibration_analysis import vibration_analysis


def test_native_process_signal_preserves_historical_pipeline() -> None:
    fs = 2048.0
    time = np.arange(1024, dtype=np.float64) / fs
    signal = 2.0 + np.sin(2.0 * np.pi * 80.0 * time)

    time_signal, frequency, ordinary, envelope = process_signal(signal, fs, "native")
    expected_frequency, expected_ordinary, expected_envelope = native_spectra(signal, fs)

    np.testing.assert_array_equal(time_signal, signal)
    np.testing.assert_array_equal(frequency, expected_frequency)
    np.testing.assert_array_equal(ordinary, expected_ordinary)
    np.testing.assert_array_equal(envelope, expected_envelope)


def test_vibration_process_signal_uses_rms_spectra_and_business_cutoff() -> None:
    fs = 2048.0
    time = np.arange(1024, dtype=np.float64) / fs
    signal = 3.0 + 2.0 * np.sin(2.0 * np.pi * 96.0 * time)

    time_signal, frequency, ordinary, envelope = process_signal(
        signal, fs, "vibration_analysis"
    )
    expected = vibration_analysis(signal, fs)
    keep = expected["frequency"] <= fs / 2.56

    np.testing.assert_allclose(time_signal, expected["time_signal"])
    np.testing.assert_array_equal(frequency, expected["frequency"][keep])
    np.testing.assert_allclose(ordinary, expected["rms_spectrum"][keep])
    np.testing.assert_allclose(envelope, expected["envelope_rms_spectrum"][keep])
    np.testing.assert_allclose(
        process_time_signal(signal, "vibration_analysis"), expected["time_signal"]
    )


def test_load_waveform_detects_openxml_payload_with_csv_suffix(tmp_path) -> None:
    workbook_path = tmp_path / "waveform.csv"
    worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1"><v>1.25</v></c><c r="B1"><v>99</v></c></row>
    <row r="2"><c r="A2"><v>-2.5</v></c><c r="B2"><v>98</v></c></row>
    <row r="3"><c r="A3"><v>3.75</v></c><c r="B3"><v>97</v></c></row>
  </sheetData>
</worksheet>"""
    with ZipFile(workbook_path, "w") as workbook:
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet)

    np.testing.assert_allclose(load_waveform(workbook_path), [1.25, -2.5, 3.75])


def test_preprocess_state_round_trip_and_legacy_default(tmp_path) -> None:
    grid = FrequencyGrid(np.array([0.0, 1.0]), 1.0, 1.0)
    state = PreprocessState(2.0, grid, 600.0, 1900.0, "vibration_analysis")
    scaler = MechanismScaler(np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32))
    save_preprocess_state(tmp_path, state, scaler)

    loaded, _ = load_preprocess_state(tmp_path)
    assert loaded.signal_processing_mode == "vibration_analysis"

    metadata_path = tmp_path / "preprocess.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("signal_processing_mode")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    legacy, _ = load_preprocess_state(tmp_path)
    assert legacy.signal_processing_mode == "native"


def test_model_config_validates_signal_processing_mode() -> None:
    assert (
        ModelConfig("v1", "dfig", signal_processing_mode="vibration_analysis")
        .signal_processing_mode
        == "vibration_analysis"
    )
    with pytest.raises(ValueError, match="unsupported signal_processing_mode"):
        ModelConfig("v1", "dfig", signal_processing_mode="unknown")  # type: ignore[arg-type]
