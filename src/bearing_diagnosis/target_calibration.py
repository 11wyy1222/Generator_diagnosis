from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import binary_metrics
from .freeze import verify_freeze_manifest
from .schemas import SampleRecord, read_jsonl


def _score_metrics(
    targets: list[int], scores: list[float], threshold: float, *, strict: bool = False,
) -> dict[str, Any]:
    labels = np.asarray(targets, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    predicted = values > threshold if strict else values >= threshold
    if np.unique(labels).size == 2:
        effective = float(np.nextafter(threshold, np.inf)) if strict else threshold
        metrics = binary_metrics(labels, values, effective)
        metrics["threshold"] = threshold
        metrics["threshold_comparison"] = ">" if strict else ">="
        return metrics
    if np.all(labels == 0):
        return {
            "threshold": threshold,
            "threshold_comparison": ">" if strict else ">=",
            "sample_count": len(labels),
            "specificity_range_label": float(np.mean(~predicted)),
            "false_positive_rate_range_label": float(np.mean(predicted)),
            "true_negative_count": int(np.sum(~predicted)),
            "false_positive_count": int(np.sum(predicted)),
            "metrics_scope": "normal_side_only",
        }
    return {
        "threshold": threshold,
        "threshold_comparison": ">" if strict else ">=",
        "sample_count": len(labels),
        "recall_range_label": float(np.mean(predicted)),
        "false_negative_count": int(np.sum(~predicted)),
        "true_positive_count": int(np.sum(predicted)),
        "metrics_scope": "abnormal_side_only",
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rpm = np.asarray([float(row["rpm"]) for row in rows], dtype=np.float64)
    times = [row["acquisition_time"] for row in rows]
    return {
        "sample_count": len(rows),
        "first_time": min(times).isoformat(sep=" "),
        "last_time": max(times).isoformat(sep=" "),
        "rpm_min": float(np.min(rpm)),
        "rpm_p10": float(np.quantile(rpm, .10)),
        "rpm_p50": float(np.quantile(rpm, .50)),
        "rpm_p90": float(np.quantile(rpm, .90)),
        "rpm_max": float(np.max(rpm)),
        "by_sensor": dict(Counter(str(row["sensor_position"]) for row in rows)),
    }


def _score_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logits = np.asarray([float(row["abnormal_logit"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray(
        [float(row["abnormal_probability"]) for row in rows], dtype=np.float64
    )
    values, counts = np.unique(logits, return_counts=True)
    largest_tie = int(np.max(counts))
    return {
        "logit_min": float(np.min(logits)),
        "logit_p50": float(np.quantile(logits, .50)),
        "logit_p95": float(np.quantile(logits, .95, method="higher")),
        "logit_p99": float(np.quantile(logits, .99, method="higher")),
        "logit_max": float(np.max(logits)),
        "logit_unique_count": int(values.size),
        "largest_logit_tie_count": largest_tie,
        "largest_logit_tie_rate": largest_tie / len(logits),
        "probability_exact_one_count": int(np.sum(probabilities == 1.0)),
        "probability_exact_one_rate": float(np.mean(probabilities == 1.0)),
        "saturated_probability_logit_unique_count": int(np.unique(
            logits[probabilities == 1.0]
        ).size) if np.any(probabilities == 1.0) else 0,
    }


def evaluate_target_calibration(
    run_dir: str | Path,
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    object_id: str,
    *,
    normal_fraction: float = .60,
    normal_quantile: float = .95,
    embargo_hours: float = 24.0,
    freeze_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if not 0 < normal_fraction < 1:
        raise ValueError("normal_fraction must be in (0, 1)")
    if not 0 < normal_quantile < 1:
        raise ValueError("normal_quantile must be in (0, 1)")
    if embargo_hours < 0:
        raise ValueError("embargo_hours must be non-negative")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow and PyTorch are required") from exc

    output = Path(output_dir)
    freeze_manifest = (
        verify_freeze_manifest(freeze_manifest_path) if freeze_manifest_path is not None else None
    )
    records = [SampleRecord.from_dict(raw) for raw in read_jsonl(manifest_path)]
    records = [record for record in records if record.object_id == object_id]
    if not records:
        raise ValueError(f"object_id not found in manifest: {object_id}")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("duplicate sample_id found in target object manifest")
    metadata = {record.sample_id: record for record in records}

    prediction_rows = pq.read_table(predictions_path).to_pylist()
    selected = [row for row in prediction_rows if str(row["sample_id"]) in metadata]
    if len(selected) != len(records):
        present = {str(row["sample_id"]) for row in selected}
        raise ValueError(
            f"prediction coverage mismatch: expected={len(records)} actual={len(selected)} "
            f"missing={len(set(metadata) - present)}"
        )
    if len({str(row["sample_id"]) for row in selected}) != len(selected):
        raise ValueError("duplicate sample_id found in predictions")
    if any("abnormal_logit" not in row for row in selected):
        raise ValueError(
            "predictions do not contain abnormal_logit; rerun frozen v1 inference before calibration"
        )

    rows: list[dict[str, Any]] = []
    for prediction in selected:
        record = metadata[str(prediction["sample_id"])]
        rows.append({
            **prediction,
            "object_id": record.object_id,
            "sensor_position": record.sensor_position,
            "acquisition_time": record.acquisition_time,
            "rpm": record.rpm,
            "label_source": record.label_source,
            "target": int(record.is_observed_scope_abnormal),
        })
    rows.sort(key=lambda row: (row["acquisition_time"], row["sensor_position"], str(row["sample_id"])))

    normal = [row for row in rows if int(row["target"]) == 0]
    if len(normal) < 2:
        raise ValueError("at least two confirmed-normal target samples are required")
    accepted_normal_sources = {"normal_time_range", "confirmed_normal_directory"}
    non_confirmed = [
        row for row in normal if str(row["label_source"]) not in accepted_normal_sources
    ]
    if non_confirmed:
        raise ValueError(
            "target calibration requires label_source=normal_time_range or "
            "confirmed_normal_directory"
        )
    normal_times = sorted({row["acquisition_time"] for row in normal})
    cut_index = max(0, min(len(normal_times) - 2, int(np.ceil(len(normal_times) * normal_fraction)) - 1))
    cutoff = normal_times[cut_index]
    evaluation_start = cutoff + timedelta(hours=embargo_hours)
    calibration = [row for row in normal if row["acquisition_time"] <= cutoff]
    embargo = [
        row for row in normal
        if cutoff < row["acquisition_time"] <= evaluation_start
    ]
    evaluation_normal = [row for row in normal if row["acquisition_time"] > evaluation_start]
    evaluation_abnormal = [
        row for row in rows
        if int(row["target"]) == 1 and row["acquisition_time"] > evaluation_start
    ]
    earlier_abnormal = [
        row for row in rows
        if int(row["target"]) == 1 and row["acquisition_time"] <= evaluation_start
    ]
    if earlier_abnormal:
        raise ValueError("abnormal samples occur before the post-embargo evaluation boundary")
    evaluation = evaluation_normal + evaluation_abnormal
    evaluation.sort(key=lambda row: (row["acquisition_time"], row["sensor_position"], str(row["sample_id"])))
    if not evaluation:
        raise ValueError("no post-calibration evaluation samples remain")

    calibration_ids = {str(row["sample_id"]) for row in calibration}
    evaluation_ids = {str(row["sample_id"]) for row in evaluation}
    if calibration_ids & evaluation_ids:
        raise AssertionError("sample leakage between calibration and evaluation")
    calibration_scores = np.asarray(
        [float(row["abnormal_logit"]) for row in calibration], dtype=np.float64
    )
    threshold = float(np.quantile(calibration_scores, normal_quantile, method="higher"))
    checkpoint = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    frozen_threshold = float(checkpoint["evaluation_threshold"])

    role_by_id = {
        **{str(row["sample_id"]): "normal_calibration" for row in calibration},
        **{str(row["sample_id"]): "embargo_excluded" for row in embargo},
        **{str(row["sample_id"]): "normal_evaluation" for row in evaluation_normal},
        **{str(row["sample_id"]): "abnormal_evaluation" for row in evaluation_abnormal},
    }
    output_rows = [{
        **row,
        "acquisition_time": row["acquisition_time"].isoformat(sep=" "),
        "target_calibration_role": role_by_id[str(row["sample_id"])],
        "target_threshold": threshold,
        "target_predicted_abnormal": float(row["abnormal_logit"]) > threshold,
    } for row in rows]

    targets = [int(row["target"]) for row in evaluation]
    scores = [float(row["abnormal_logit"]) for row in evaluation]
    calibrated_metrics = _score_metrics(targets, scores, threshold, strict=True)
    frozen_metrics = _score_metrics(
        targets,
        [float(row["abnormal_probability"]) for row in evaluation],
        frozen_threshold,
    )
    sensor_metrics: dict[str, Any] = {}
    for sensor in sorted({str(row["sensor_position"]) for row in evaluation}):
        sensor_rows = [row for row in evaluation if str(row["sensor_position"]) == sensor]
        sensor_metrics[sensor] = _score_metrics(
            [int(row["target"]) for row in sensor_rows],
            [float(row["abnormal_logit"]) for row in sensor_rows],
            threshold,
            strict=True,
        )

    calibration_rpm = np.asarray([float(row["rpm"]) for row in calibration])
    evaluation_rpm = np.asarray([float(row["rpm"]) for row in evaluation])
    report: dict[str, Any] = {
        "protocol": {
            "model": "v1_single_waveform",
            "score": "raw abnormal_logit",
            "threshold_method": "higher empirical quantile of earliest confirmed-normal logits",
            "normal_fraction": normal_fraction,
            "normal_quantile": normal_quantile,
            "embargo_hours": embargo_hours,
            "threshold_comparison": "logit > threshold",
            "weights_changed": False,
            "test_labels_used_for_threshold": False,
        },
        "object_id": object_id,
        "freeze_manifest": str(Path(freeze_manifest_path).resolve()) if freeze_manifest_path else None,
        "freeze_verified": freeze_manifest is not None,
        "source_model_threshold": frozen_threshold,
        "target_threshold": threshold,
        "calibration_cutoff": cutoff.isoformat(sep=" "),
        "evaluation_start_exclusive": evaluation_start.isoformat(sep=" "),
        "calibration": _coverage(calibration),
        "calibration_score_diagnostics": _score_diagnostics(calibration),
        "embargo_excluded_count": len(embargo),
        "evaluation": _coverage(evaluation),
        "evaluation_score_diagnostics": _score_diagnostics(evaluation),
        "evaluation_normal_count": len(evaluation_normal),
        "evaluation_abnormal_count": len(evaluation_abnormal),
        "rpm_outside_calibration_range_rate": float(np.mean(
            (evaluation_rpm < np.min(calibration_rpm)) | (evaluation_rpm > np.max(calibration_rpm))
        )),
        "calibration_false_positive_rate": float(np.mean(calibration_scores > threshold)),
        "source_threshold_metrics_on_same_evaluation": frozen_metrics,
        "target_calibrated_metrics": calibrated_metrics,
        "target_calibrated_metrics_by_sensor": sensor_metrics,
        "leakage_audit": {
            "duplicate_manifest_sample_ids": 0,
            "duplicate_prediction_sample_ids": 0,
            "calibration_evaluation_sample_id_overlap": 0,
            "time_order_strict": max(row["acquisition_time"] for row in calibration)
            < min(row["acquisition_time"] for row in evaluation),
        },
        "interpretation_limit": (
            "normal-only evaluation; estimates false-positive control only"
            if not evaluation_abnormal else
            "contains held-out normal and later abnormal ranges; evaluates false positives and abnormal recall"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(output_rows), output / "predictions_target_calibrated.parquet")
    (output / "metrics_target_calibrated.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"output_dir": str(output), "metrics": report}
