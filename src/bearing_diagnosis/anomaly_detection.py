from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluation import binary_metrics
from .mechanism import extract_mechanism_features
from .preprocessing import load_waveform, native_spectra, validate_waveform
from .schemas import SampleRecord

EPS = 1e-12


@dataclass
class NormalReference:
    center: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    precision: np.ndarray
    scores: np.ndarray


def extract_features(record: SampleRecord, bands: int = 16) -> np.ndarray:
    signal = load_waveform(record.waveform_path)
    reasons = validate_waveform(signal, record.waveform_length)
    if reasons:
        raise ValueError(f"rejected waveform {record.sample_id}: {', '.join(reasons)}")
    x = signal.astype(np.float64) - float(np.mean(signal))
    absolute = np.abs(x)
    rms = max(float(np.sqrt(np.mean(x * x))), EPS)
    abs_mean = max(float(np.mean(absolute)), EPS)
    peak = max(float(np.max(absolute)), EPS)
    std = max(float(np.std(x)), EPS)
    quantiles = np.quantile(absolute, [.25, .5, .75, .9, .99])
    time = [
        np.log(rms), np.log(abs_mean), np.log(peak), np.log(max(float(np.ptp(x)), EPS)),
        float(np.mean((x / std) ** 3)), float(np.mean((x / std) ** 4)),
        peak / rms, peak / abs_mean,
        float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))),
        *np.log(np.maximum(quantiles, EPS)).tolist(),
    ]
    frequency, ordinary, envelope = native_spectra(x, record.sampling_rate_hz)
    f_max = min(5000.0, float(frequency[-1]))
    edges = np.linspace(0.0, f_max, bands + 1)
    spectral: list[float] = []
    for values in (ordinary, envelope):
        selected = frequency <= f_max
        f = frequency[selected]
        s = np.maximum(values[selected].astype(np.float64), EPS)
        total = max(float(np.sum(s)), EPS)
        centroid = float(np.sum(f * s) / total) / max(f_max, EPS)
        spread = float(np.sqrt(np.sum(((f / max(f_max, EPS) - centroid) ** 2) * s) / total))
        spectral.extend([np.log(total), centroid, spread])
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (f >= left) & (f < right if right < f_max else f <= right)
            band = s[mask]
            spectral.extend([np.log(max(float(np.mean(band)), EPS)), np.log(max(float(np.max(band)), EPS))])
    mechanism: list[float] = []
    if record.component_orders is not None:
        evidence = extract_mechanism_features(
            frequency, ordinary, envelope, record.rpm, record.component_orders,
            record.sampling_rate_hz, record.waveform_length,
        )
        for values, mask in zip(evidence.features, evidence.valid_mask):
            boolean_mask = np.asarray(mask, dtype=bool)
            valid = values[boolean_mask]
            mechanism.extend([
                float(np.median(valid)) if valid.size else 0.0,
                float(np.mean(valid)) if valid.size else 0.0,
                float(np.max(valid)) if valid.size else 0.0,
                float(np.mean(boolean_mask)),
            ])
        mechanism.append(float(evidence.q_global))
    else:
        mechanism = [0.0] * 17
    return np.asarray(time + spectral + mechanism, dtype=np.float64)


def fit_reference(features: np.ndarray) -> NormalReference:
    center = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    z = np.clip((features - center) / scale, -20.0, 20.0)
    mean = np.mean(z, axis=0)
    covariance = np.cov(z, rowvar=False)
    covariance = .25 * covariance + .75 * np.diag(np.diag(covariance))
    covariance += np.eye(covariance.shape[0]) * 1e-3
    precision = np.linalg.pinv(covariance, hermitian=True)
    delta = z - mean
    scores = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return NormalReference(center, scale, mean, precision, np.sort(scores))


def score_features(features: np.ndarray, reference: NormalReference) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip((features - reference.center) / reference.scale, -20.0, 20.0)
    delta = z - reference.mean
    raw = np.einsum("ij,jk,ik->i", delta, reference.precision, delta)
    percentile = np.searchsorted(reference.scores, raw, side="right") / (reference.scores.size + 1.0)
    return raw, percentile


def _extract(records: Sequence[SampleRecord], label: str) -> np.ndarray:
    rows = []
    for index, record in enumerate(records, 1):
        rows.append(extract_features(record))
        if index % 50 == 0 or index == len(records):
            print(f"[anomaly-features] {label} processed={index}/{len(records)}", flush=True)
    return np.stack(rows)


def _fit_grouped(records: Sequence[SampleRecord], features: np.ndarray) -> tuple[dict[str, NormalReference], dict[str, str]]:
    references = {"__global__": fit_reference(features)}
    mapping: dict[str, str] = {}
    for position in sorted({record.sensor_position for record in records}):
        indices = [index for index, record in enumerate(records) if record.sensor_position == position]
        key = f"position:{position}"
        if len(indices) >= 30:
            references[key] = fit_reference(features[indices])
            mapping[position] = key
        else:
            mapping[position] = "__global__"
    return references, mapping


def _predict(records: Sequence[SampleRecord], features: np.ndarray, references: dict[str, NormalReference], mapping: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for index, record in enumerate(records):
        key = mapping.get(record.sensor_position, "__global__")
        raw, probability = score_features(features[index:index + 1], references[key])
        rows.append({
            "sample_id": record.sample_id, "object_id": record.object_id,
            "sensor_position": record.sensor_position, "rpm": record.rpm,
            "rpm_bin": record.rpm_bin, "range_id": record.range_id,
            "range_position": record.range_position,
            "is_observed_scope_abnormal": bool(record.is_observed_scope_abnormal),
            "reference_key": key, "anomaly_score": float(raw[0]),
            "abnormal_probability": float(probability[0]),
        })
    return rows


def _fold(record: SampleRecord, folds: int = 5) -> int:
    key = f"{record.object_id}|{record.sample_group_id or record.sample_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % folds


def _cross_validated_normal_probabilities(records: Sequence[SampleRecord], features: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(len(records), dtype=np.float64)
    fold_ids = np.asarray([_fold(record) for record in records])
    for fold in range(5):
        fit_indices = np.flatnonzero(fold_ids != fold)
        score_indices = np.flatnonzero(fold_ids == fold)
        references, mapping = _fit_grouped([records[i] for i in fit_indices], features[fit_indices])
        rows = _predict([records[i] for i in score_indices], features[score_indices], references, mapping)
        probabilities[score_indices] = [float(row["abnormal_probability"]) for row in rows]
    return probabilities


def _metrics(predictions: Sequence[dict[str, object]], threshold: float) -> dict[str, object]:
    targets = np.asarray([int(row["is_observed_scope_abnormal"]) for row in predictions])
    probabilities = np.asarray([float(row["abnormal_probability"]) for row in predictions])
    predicted = probabilities >= threshold
    result: dict[str, object] = {
        "sample_count": len(predictions), "threshold": threshold,
        "predicted_abnormal_count": int(np.sum(predicted)),
        "probability_quantiles": {name: float(np.quantile(probabilities, q)) for name, q in (("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9))},
    }
    if np.unique(targets).size == 2:
        result.update(binary_metrics(targets, probabilities, threshold))
    elif targets[0] == 1:
        result.update({"recall_range_label": float(np.mean(predicted)), "metrics_scope": "abnormal_side_only"})
    else:
        fpr = float(np.mean(predicted))
        result.update({"specificity_range_label": 1.0 - fpr, "false_positive_rate_range_label": fpr, "metrics_scope": "normal_side_only"})
    return result


def _write_predictions(path: Path, rows: Sequence[dict[str, object]], threshold: float) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist([dict(row, predicted_abnormal=float(row["abnormal_probability"]) >= threshold) for row in rows]), path)


def _save_model(output: Path, references: dict[str, NormalReference], mapping: dict[str, str], threshold: float) -> None:
    arrays: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}
    for index, (key, reference) in enumerate(references.items()):
        prefix = f"g{index}"
        groups[key] = prefix
        for name in ("center", "scale", "mean", "precision", "scores"):
            arrays[f"{prefix}_{name}"] = getattr(reference, name)
    np.savez_compressed(output / "normal_reference.npz", **arrays)
    metadata = {"model_name": "anomaly_dfig_robust_cov_v1", "threshold": threshold, "groups": groups,
                "position_mapping": mapping, "rpm_used_as_feature": False, "normal_cv_folds": 5}
    (output / "anomaly_model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model(run_dir: Path) -> tuple[dict[str, NormalReference], dict[str, str], float]:
    metadata = json.loads((run_dir / "anomaly_model.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "normal_reference.npz")
    references = {}
    for key, prefix in metadata["groups"].items():
        references[key] = NormalReference(*(data[f"{prefix}_{name}"] for name in ("center", "scale", "mean", "precision", "scores")))
    return references, metadata["position_mapping"], float(metadata["threshold"])


def train_anomaly(development_records: Sequence[SampleRecord], output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    normal = [record for record in development_records if not record.is_observed_scope_abnormal]
    abnormal = [record for record in development_records if record.is_observed_scope_abnormal]
    print(f"[anomaly-setup] normal_train={len(normal)} abnormal_validation={len(abnormal)}", flush=True)
    normal_features = _extract(normal, "normal-train")
    cv_probability = _cross_validated_normal_probabilities(normal, normal_features)
    threshold = float(np.quantile(cv_probability, .95))
    references, mapping = _fit_grouped(normal, normal_features)
    abnormal_features = _extract(abnormal, "abnormal-validation")
    predictions = _predict(abnormal, abnormal_features, references, mapping)
    metrics = _metrics(predictions, threshold)
    metrics.update({"normal_train_count": len(normal), "normal_cv_false_positive_rate": float(np.mean(cv_probability >= threshold)), "abnormal_validation_count": len(abnormal)})
    _save_model(output, references, mapping, threshold)
    _write_predictions(output / "predictions_validation.parquet", predictions, threshold)
    (output / "metrics_validation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "anomaly_split.jsonl").open("w", encoding="utf-8") as handle:
        for record in development_records:
            role = "normal_train" if not record.is_observed_scope_abnormal else "abnormal_validation"
            handle.write(json.dumps({"sample_id": record.sample_id, "original_split": record.dataset_split, "anomaly_role": role, "normal_cv_fold": _fold(record) if role == "normal_train" else None}, ensure_ascii=False) + "\n")
    print(f"[anomaly-done] threshold={threshold:.6f} validation_recall={metrics['recall_range_label']:.4f}", flush=True)
    return {"run_dir": str(output), "threshold": threshold, "metrics": metrics}


def test_anomaly(run_dir: str | Path, records: Sequence[SampleRecord], output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    references, mapping, threshold = _load_model(Path(run_dir))
    print(f"[anomaly-test-setup] samples={len(records)} threshold={threshold:.6f}", flush=True)
    predictions = _predict(records, _extract(records, "test"), references, mapping)
    metrics = _metrics(predictions, threshold)
    _write_predictions(output / "predictions_test.parquet", predictions, threshold)
    (output / "metrics_test.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[anomaly-test-done] predicted_abnormal={metrics['predicted_abnormal_count']}/{len(records)}", flush=True)
    return {"output_dir": str(output), "metrics": metrics}
