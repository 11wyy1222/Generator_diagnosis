from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .artifacts import load_preprocess_state
from .config import ModelConfig
from .dataset import BearingDataset, LengthBatchSampler
from .evaluation import binary_metrics
from .model import BearingDiagnosisModel
from .schemas import SampleRecord
from .training import collate_same_length, evaluate


def load_run_config(run_dir: str | Path) -> ModelConfig:
    checkpoint = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    raw = dict(checkpoint["config"])
    raw["random_seeds"] = tuple(raw["random_seeds"])
    return ModelConfig(**raw)


def _probability_summary(
    rows: Iterable[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    rows = list(rows)
    probabilities = np.asarray([float(row["abnormal_probability"]) for row in rows])
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "threshold": threshold,
        "predicted_abnormal_count": int(np.sum(probabilities >= threshold)),
        "probability_quantiles": {
            f"p{percentile}": float(np.percentile(probabilities, percentile))
            for percentile in (10, 25, 50, 75, 90)
        },
    }
    if np.unique(targets).size == 2:
        summary.update(binary_metrics(targets, probabilities, threshold))
    elif targets.size and np.all(targets == 1):
        summary["recall_range_label"] = float(np.mean(probabilities >= threshold))
        summary["metrics_scope"] = "abnormal_side_only"
    elif targets.size and np.all(targets == 0):
        summary["specificity_range_label"] = float(np.mean(probabilities < threshold))
        summary["false_positive_rate_range_label"] = float(np.mean(probabilities >= threshold))
        summary["metrics_scope"] = "normal_side_only"
    return summary


def _group_summaries(
    rows: list[dict[str, Any]], field: str, threshold: float
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: _probability_summary(group, threshold)
        for key, group in sorted(groups.items())
    }


def test_one_run(
    run_dir: str | Path,
    records: list[SampleRecord],
    output_dir: str | Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("the requested test split has no records for this model machine type")
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = load_run_config(run_dir)
    if {record.machine_type for record in records} != {config.machine_type}:
        raise ValueError("test manifest machine_type does not match the trained model")
    missing_orders = sorted({record.object_id for record in records if record.component_orders is None})
    if missing_orders and config.experiment != "spectrum_only":
        raise ValueError(
            "component orders are unconfirmed for " + ", ".join(missing_orders)
            + "; evaluate these records only with a spectrum-only model"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BearingDiagnosisModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    threshold = float(checkpoint["evaluation_threshold"])
    preprocess, scaler = load_preprocess_state(run_dir)
    loader = DataLoader(
        BearingDataset(
            records,
            preprocess,
            scaler,
            allow_missing_mechanism=config.experiment == "spectrum_only",
        ),
        batch_sampler=LengthBatchSampler(records, config.batch_size),
        collate_fn=collate_same_length,
    )
    print(
        f"[test-setup] model={config.model_name} split={records[0].dataset_split} "
        f"samples={len(records)} threshold={threshold:.6f} device={device}",
        flush=True,
    )
    predictions, _ = evaluate(
        model,
        loader,
        device,
        config.mechanism_aux_weight,
        progress_every=50,
    )
    metadata = {record.sample_id: record for record in records}
    enriched: list[dict[str, Any]] = []
    for prediction in predictions:
        record = metadata[str(prediction["sample_id"])]
        enriched.append({
            **prediction,
            "object_id": record.object_id,
            "sensor_position": record.sensor_position,
            "acquisition_time": record.acquisition_time.isoformat(sep=" "),
            "rpm": record.rpm,
            "rpm_bin": record.rpm_bin,
            "range_id": record.range_id,
            "range_position": record.range_position,
            "dataset_split": record.dataset_split,
        })

    overall = _probability_summary(enriched, threshold)
    metrics: dict[str, Any] = {
        "model_name": config.model_name,
        "machine_type": config.machine_type,
        "dataset_split": records[0].dataset_split,
        "evaluation_threshold_source": "frozen_development_validation",
        "overall": overall,
        "by_sensor_position": _group_summaries(enriched, "sensor_position", threshold),
        "by_object_id": _group_summaries(enriched, "object_id", threshold),
        "by_rpm_bin": _group_summaries(enriched, "rpm_bin", threshold),
        "by_range_id": _group_summaries(enriched, "range_id", threshold),
        "by_range_position": _group_summaries(
            [row for row in enriched if int(row["target"]) == 1],
            "range_position",
            threshold,
        ),
    }
    if config.machine_type == "dfig":
        metrics["by_rpm_scope"] = {
            "in_range_test": _probability_summary(
                [row for row in enriched if float(row["rpm"]) <= config.rpm_max], threshold
            ),
            "rpm_ood_test": _probability_summary(
                [row for row in enriched if float(row["rpm"]) > config.rpm_max], threshold
            ),
        }

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required; run: pip install -r requirements.txt") from exc
    pq.write_table(pa.Table.from_pylist(enriched), output_dir / "predictions_test.parquet")
    (output_dir / "metrics_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[test-done] samples={len(records)} output_dir={output_dir} "
        f"median_probability={overall['probability_quantiles']['p50']:.4f}",
        flush=True,
    )
    return {"output_dir": str(output_dir), "metrics": metrics}
