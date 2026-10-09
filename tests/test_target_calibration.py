from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bearing_diagnosis.schemas import SampleRecord, write_jsonl
from bearing_diagnosis.target_calibration import evaluate_target_calibration


def test_early_normal_calibration_and_later_evaluation_are_disjoint(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    torch = pytest.importorskip("torch")
    start = datetime(2026, 1, 1)
    scores = [.10, .20, .15, .05, .12, .18, .14, .10, .15, .19, .80, .90]
    records = []
    predictions = []
    for index, score in enumerate(scores):
        abnormal = index >= 10
        record = SampleRecord(
            sample_id=f"sample-{index}", object_id="TARGET", project_id="project",
            turbine_id="1", machine_type="dfig", sensor_position="驱动端",
            waveform_path=f"wave-{index}.csv", acquisition_time=start + timedelta(days=index),
            sampling_rate_hz=25600.0, waveform_length=131072, rpm=1000.0 + index,
            rpm_source="filename", range_id="abnormal" if abnormal else "normal",
            fault_event_id="event" if abnormal else None,
            is_observed_scope_abnormal=abnormal,
            label_source="abnormal_time_range" if abnormal else "normal_time_range",
            component_orders=None, dataset_split="cross_turbine_test",
        )
        records.append(record.to_dict())
        predictions.append({
            "sample_id": record.sample_id,
            "abnormal_logit": score,
            "abnormal_probability": score,
        })

    manifest = tmp_path / "manifest.jsonl"
    prediction_path = tmp_path / "predictions.parquet"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "result"
    write_jsonl(manifest, records)
    pq.write_table(pa.Table.from_pylist(predictions), prediction_path)
    run_dir.mkdir()
    torch.save({"evaluation_threshold": .5}, run_dir / "model.pt")

    result = evaluate_target_calibration(
        run_dir, manifest, prediction_path, output_dir, "TARGET",
        normal_fraction=.60, normal_quantile=.95, embargo_hours=24.0,
    )["metrics"]

    assert result["target_threshold"] == pytest.approx(.20)
    assert result["calibration"]["sample_count"] == 6
    assert result["embargo_excluded_count"] == 1
    assert result["evaluation_normal_count"] == 3
    assert result["evaluation_abnormal_count"] == 2
    assert result["leakage_audit"]["calibration_evaluation_sample_id_overlap"] == 0
    assert result["leakage_audit"]["time_order_strict"] is True
    assert result["target_calibrated_metrics"]["recall_range_label"] == pytest.approx(1.0)


def test_target_calibration_rejects_non_confirmed_normal(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    torch = pytest.importorskip("torch")
    start = datetime(2026, 1, 1)
    records = []
    predictions = []
    for index in range(3):
        record = SampleRecord(
            sample_id=f"sample-{index}", object_id="TARGET", project_id="project",
            turbine_id="1", machine_type="dfig", sensor_position="驱动端",
            waveform_path=f"wave-{index}.csv", acquisition_time=start + timedelta(days=index),
            sampling_rate_hz=25600.0, waveform_length=131072, rpm=1000.0,
            rpm_source="filename", range_id="normal", fault_event_id=None,
            is_observed_scope_abnormal=False, label_source="abnormal_time_range",
            component_orders=None, dataset_split="external_normal_test",
        )
        records.append(record.to_dict())
        predictions.append({
            "sample_id": record.sample_id,
            "abnormal_logit": .1,
            "abnormal_probability": .1,
        })
    manifest = tmp_path / "manifest.jsonl"
    prediction_path = tmp_path / "predictions.parquet"
    run_dir = tmp_path / "run"
    write_jsonl(manifest, records)
    pq.write_table(pa.Table.from_pylist(predictions), prediction_path)
    run_dir.mkdir()
    torch.save({"evaluation_threshold": .5}, run_dir / "model.pt")

    with pytest.raises(ValueError, match="normal_time_range"):
        evaluate_target_calibration(run_dir, manifest, prediction_path, tmp_path / "out", "TARGET")
