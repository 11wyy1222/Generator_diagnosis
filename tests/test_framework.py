from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest

from bearing_diagnosis.config import ModelConfig
from bearing_diagnosis.evaluation import select_f1_threshold
from bearing_diagnosis.mechanism import FEATURE_COUNT, extract_mechanism_features
from bearing_diagnosis.preprocessing import (
    FrequencyGrid,
    PreprocessState,
    load_waveform,
    native_spectra,
    validate_waveform,
)
from bearing_diagnosis.schemas import SampleRecord, parse_waveform_filename
from bearing_diagnosis.splitting import assert_no_group_leakage, split_development_records
from bearing_diagnosis.training import checkpoint_improved, update_convergence_count


ORDERS = {"outer_race": 13.156, "inner_race": 14.844, "rolling_element": 8.2638, "cage": 0.4698}


def test_checkpoint_selection_uses_validation_loss_for_pr_auc_ties() -> None:
    assert checkpoint_improved(0.99, 0.6, 0.98, 0.1)
    assert checkpoint_improved(1.0, 0.2, 1.0, 0.3)
    assert not checkpoint_improved(1.0, 0.4, 1.0, 0.3)
    assert not checkpoint_improved(0.999, 0.01, 1.0, 0.3)


def test_convergence_count_requires_consecutive_losses_at_or_below_threshold() -> None:
    count = update_convergence_count(0.01, 0.01, 0)
    assert count == 1
    count = update_convergence_count(0.009, 0.01, count)
    assert count == 2
    assert update_convergence_count(0.011, 0.01, count) == 0


def record(day: int, abnormal: bool, position: str = "3点") -> SampleRecord:
    time = datetime(2026, 1, 1) + timedelta(days=day)
    return SampleRecord(
        sample_id=f"s-{abnormal}-{day}-{position}", object_id="SD_DEV_23", project_id="p",
        turbine_id="23", machine_type="semi_direct", sensor_position=position,
        waveform_path="unused.npy", acquisition_time=time, sampling_rate_hz=25600.0,
        waveform_length=4096, rpm=220.0, rpm_source="filename", range_id="abnormal" if abnormal else "normal",
        fault_event_id="event-1" if abnormal else None,
        is_observed_scope_abnormal=abnormal,
        label_source="abnormal_time_range" if abnormal else "normal_time_range",
        component_orders=ORDERS, sample_group_id=f"group-{abnormal}-{day}",
        range_position="early" if abnormal else None,
    )


def test_filename_parser_uses_right_suffix_and_exact_position() -> None:
    parsed = parse_waveform_filename(
        "广东_23#_发电机3点径向_25600Hz_215.5_20231206051000.csv", require_position=True
    )
    assert parsed.sensor_position == "3点"
    assert parsed.sampling_rate_hz == 25600.0
    assert parsed.rpm == 215.5
    assert parsed.acquisition_time == datetime(2023, 12, 6, 5, 10)


def test_filename_parser_accepts_sampling_rate_without_hz() -> None:
    parsed = parse_waveform_filename(
        "广东_22#_发电机6点径向_25600_215.5_20250810145910.csv",
        require_position=True,
    )
    assert parsed.sensor_position == "6点"
    assert parsed.sampling_rate_hz == 25600.0


def test_filename_parser_accepts_actual_dfig_suffix_variants() -> None:
    upper = parse_waveform_filename(
        "项目_41#风机_发电机自由端9V_25600HZ_610RPM_20260415133600.csv"
    )
    decimal = parse_waveform_filename(
        "项目_FJ14_发电机驱动端径向_25600.0Hz_1000.5RPM_20260626161050.csv"
    )
    assert upper.sampling_rate_hz == 25600.0
    assert upper.rpm == 610.0
    assert decimal.sampling_rate_hz == 25600.0
    assert decimal.rpm == 1000.5


def test_csv_loader_uses_only_first_column(tmp_path: Path) -> None:
    path = tmp_path / "waveform.csv"
    path.write_text("1,100,1000\n2,200,2000\n3,300,3000\n4,400,4000\n", encoding="utf-8")
    assert np.array_equal(load_waveform(path), np.array([1.0, 2.0, 3.0, 4.0]))


def test_manifest_loader_filters_split_and_machine_type(tmp_path: Path) -> None:
    from dataclasses import replace
    from bearing_diagnosis.dataset import records_from_manifest
    from bearing_diagnosis.schemas import write_jsonl

    semi_direct = replace(record(0, False), dataset_split="train")
    dfig = replace(
        record(1, False),
        sample_id="dfig-1",
        machine_type="dfig",
        dataset_split="train",
    )
    validation = replace(record(2, False), sample_id="validation", dataset_split="validation")
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, [item.to_dict() for item in (semi_direct, dfig, validation)])
    selected = records_from_manifest(path, "train", "semi_direct")
    assert [item.sample_id for item in selected] == [semi_direct.sample_id]


def test_frequency_grid_and_mechanism_shapes() -> None:
    fs, length, rpm = 2048.0, 4096, 300.0
    time = np.arange(length) / fs
    signal = np.sin(2 * np.pi * (rpm / 60 * ORDERS["outer_race"]) * time)
    frequency, ordinary, envelope = native_spectra(signal, fs)
    grid = FrequencyGrid.fit([(fs, length)], 500.0)
    assert grid.transform(frequency, ordinary, envelope).shape[0] == 2
    evidence = extract_mechanism_features(frequency, ordinary, envelope, rpm, ORDERS, fs, length)
    assert evidence.features.shape == (4, FEATURE_COUNT)
    assert evidence.valid_mask.shape == evidence.features.shape
    assert 0 <= evidence.q_global <= 1


def test_v2_order_grid_aligns_equal_orders_without_fault_order_parameters() -> None:
    fs, length = 2048.0, 4096
    grid = FrequencyGrid.fit([(fs, length)], 500.0)
    state = PreprocessState(
        1.0, grid, 180.0, 360.0, "shaft_order_v2", (1.0, 2.0, 3.0), 0.08, 0.25
    )
    peaks: list[float] = []
    for rpm in (240.0, 300.0):
        time = np.arange(length) / fs
        signal = np.sin(2 * np.pi * (rpm / 60.0) * 7.5 * time)
        frequency, ordinary, envelope = native_spectra(signal, fs)
        transformed = state.transform_spectrum(frequency, ordinary, envelope, rpm)
        assert transformed.shape == (4, grid.axis_hz.size)
        peaks.append(float(state.order_axis[np.argmax(transformed[0])]))
    assert peaks[0] == pytest.approx(7.5, abs=0.15)
    assert peaks[1] == pytest.approx(7.5, abs=0.15)


def test_v2_soft_suppression_preserves_raw_channels_and_attenuates_shaft_harmonics() -> None:
    fs, length, rpm = 2048.0, 4096, 300.0
    grid = FrequencyGrid.fit([(fs, length)], 500.0)
    state = PreprocessState(
        1.0, grid, 180.0, 350.0, "shaft_order_v2", (1.0, 2.0, 3.0), 0.08, 0.25
    )
    time = np.arange(length) / fs
    signal = np.sin(2 * np.pi * (rpm / 60.0) * time)
    frequency, ordinary, envelope = native_spectra(signal, fs)
    transformed = state.transform_spectrum(frequency, ordinary, envelope, rpm)
    shaft_bin = int(np.argmin(np.abs(state.order_axis - 1.0)))
    assert transformed[2, shaft_bin] < transformed[0, shaft_bin]
    assert transformed[3, shaft_bin] <= transformed[1, shaft_bin]


def test_waveform_admission_rejects_constant() -> None:
    assert "constant_or_fill_value" in validate_waveform(np.ones(128))


def test_group_split_has_no_leakage_and_two_training_blocks() -> None:
    records = [record(day, abnormal, position) for abnormal in (False, True) for day in range(6) for position in ("3点", "6点")]
    split = split_development_records(records)
    assert_no_group_leakage(split)
    assert {item.dataset_split for item in split} == {"train", "validation"}


def test_threshold_is_selected_from_validation_f1() -> None:
    assert select_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.5)


def test_all_ablation_models_forward_and_zero_reliability_degrades() -> None:
    torch = pytest.importorskip("torch")
    from bearing_diagnosis.model import BearingDiagnosisModel, weak_supervision_loss

    batch, length, bins = 2, 512, 256
    inputs = {
        "waveform": torch.randn(batch, 1, length),
        "spectrum": torch.rand(batch, 2, bins),
        "rpm_normalized": torch.rand(batch),
        "mechanism_features": torch.rand(batch, 4, FEATURE_COUNT),
        "mechanism_valid_mask": torch.ones(batch, 4, FEATURE_COUNT),
        "q_global": torch.tensor([0.0, 1.0]),
    }
    for experiment in ("spectrum_only", "mechanism_only", "concat", "gated"):
        model = BearingDiagnosisModel(
            ModelConfig("test", "semi_direct", experiment=experiment, time_pool_segments=2, spectrum_pool_segments=2)
        )
        output = model(**inputs)
        assert output["abnormal_logit"].shape == (batch,)
        loss, details = weak_supervision_loss(output, torch.tensor([0.0, 1.0]))
        assert torch.isfinite(loss)
        assert details["main_loss"] >= 0
        if experiment == "gated":
            assert output["g_global"][0].item() == 0.0

    v2_inputs = dict(inputs)
    v2_inputs["spectrum"] = torch.rand(batch, 4, bins)
    v2 = BearingDiagnosisModel(
        ModelConfig(
            "test-v2",
            "semi_direct",
            time_pool_segments=2,
            spectrum_pool_segments=2,
            spectrum_representation="shaft_order_v2",
            gated_fusion="competitive_v2",
        )
    )
    output = v2(**v2_inputs)
    assert torch.allclose(output["spectrum_weight"] + output["mechanism_weight"], torch.ones(batch))
    assert output["mechanism_weight"][0].item() == 0.0


def test_balanced_sampler_keeps_length_and_label_balance() -> None:
    from bearing_diagnosis.dataset import BalancedGroupBatchSampler

    records = [record(day, abnormal) for abnormal in (False, True) for day in range(6)]
    batches = list(BalancedGroupBatchSampler(records, batch_size=4))
    assert batches
    for batch in batches:
        selected = [records[index] for index in batch]
        assert len({item.waveform_length for item in selected}) == 1
        assert sum(item.is_observed_scope_abnormal for item in selected) * 2 == len(selected)


def test_balanced_sampler_equalizes_abnormal_stages() -> None:
    from dataclasses import replace
    from bearing_diagnosis.dataset import BalancedGroupBatchSampler

    records = []
    for day in range(12):
        records.append(replace(record(day, False), sample_id=f"normal-{day}"))
    offset = 100
    for stage, count in (("early", 12), ("middle", 8), ("late", 4)):
        for index in range(count):
            records.append(
                replace(
                    record(offset + index, True),
                    sample_id=f"{stage}-{index}",
                    sample_group_id=f"{stage}-group-{index}",
                    range_position=stage,
                )
            )
        offset += count
    sampler = BalancedGroupBatchSampler(records, batch_size=8)
    selected = [records[index] for batch in sampler for index in batch]
    assert sum(not item.is_observed_scope_abnormal for item in selected) == 12
    assert {
        stage: sum(item.range_position == stage for item in selected)
        for stage in ("early", "middle", "late")
    } == {"early": 4, "middle": 4, "late": 4}


def test_evaluation_sampler_never_mixes_lengths() -> None:
    from dataclasses import replace
    from bearing_diagnosis.dataset import LengthBatchSampler

    records = [record(day, bool(day % 2)) for day in range(6)]
    records = [replace(item, waveform_length=1024 if index % 2 else 2048) for index, item in enumerate(records)]
    for batch in LengthBatchSampler(records, batch_size=4):
        assert len({records[index].waveform_length for index in batch}) == 1


def test_one_epoch_training_and_inference_artifacts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    from dataclasses import replace
    from bearing_diagnosis.inference import BearingInference, InferenceInput
    from bearing_diagnosis.training import train_one_run

    fs, length = 2048.0, 512
    times = np.arange(length) / fs
    records: list[SampleRecord] = []
    for split, count in (("train", 4), ("validation", 2)):
        for abnormal in (False, True):
            for index in range(count):
                signal = np.sin(2 * np.pi * (40 if abnormal else 17) * times)
                signal += 0.02 * np.sin(2 * np.pi * (90 + index) * times)
                path = tmp_path / f"{split}-{abnormal}-{index}.npy"
                np.save(path, signal)
                records.append(
                    replace(
                        record(index, abnormal),
                        sample_id=path.stem,
                        waveform_path=str(path),
                        sampling_rate_hz=fs,
                        waveform_length=length,
                        dataset_split=split,
                        sample_group_id=f"{split}-{abnormal}-{index}",
                    )
                )
    config = ModelConfig(
        "smoke", "semi_direct", max_epochs=1, early_stopping_patience=1,
        batch_size=4, time_pool_segments=2, spectrum_pool_segments=2,
        business_f_max_hz=500.0,
        spectrum_representation="shaft_order_v2", gated_fusion="competitive_v2",
    )
    run_dir = tmp_path / "run"
    result = train_one_run(
        config,
        [item for item in records if item.dataset_split == "train"],
        [item for item in records if item.dataset_split == "validation"],
        run_dir,
        device_name="cpu",
    )
    assert 0 <= result["threshold"] <= 1
    for name in (
        "model.pt", "preprocess.json", "frequency_grid.npy", "predictions_validation.parquet",
        "model_best_during_training.pt", "gate_monitoring.parquet", "metrics_validation.json",
        "model_card.md", "data_snapshot.json",
    ):
        assert (run_dir / name).is_file()
    engine = BearingInference(run_dir)
    sample = records[0]
    external, internal = engine.predict(
        InferenceInput(sample.sample_id, sample.waveform_path, fs, sample.rpm, sample.component_orders)
    )
    assert set(external) == {"sample_id", "abnormal_probability", "component_probabilities"}
    assert external["component_probabilities"] is None
    assert 0 <= external["abnormal_probability"] <= 1
    assert "g_global" in internal
    assert internal["spectrum_weight"] + internal["mechanism_weight"] == pytest.approx(1.0)

    from bearing_diagnosis.testing import test_one_run

    test_result = test_one_run(
        run_dir,
        [item for item in records if item.dataset_split == "validation"],
        tmp_path / "test-output",
        "cpu",
    )
    assert test_result["metrics"]["overall"]["sample_count"] == 4
    assert "early" in test_result["metrics"]["by_range_position"]
    assert (tmp_path / "test-output" / "predictions_test.parquet").is_file()
    assert (tmp_path / "test-output" / "metrics_test.json").is_file()


def test_training_defaults_to_cuda_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from bearing_diagnosis import training

    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is intentionally disabled"):
        training.resolve_training_device()


def test_training_allows_explicit_cpu_for_smoke_tests() -> None:
    pytest.importorskip("torch")
    from bearing_diagnosis.training import resolve_training_device

    assert resolve_training_device("cpu").type == "cpu"
