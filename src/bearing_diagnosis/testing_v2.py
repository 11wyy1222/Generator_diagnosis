from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import load_preprocess_state
from .dataset import BearingDataset, TimeGroupBatchSampler
from .model import BearingDiagnosisModel
from .schemas import SampleRecord
from .testing import _group_summaries, _probability_summary, load_run_config
from .training import collate_same_length, evaluate


def test_one_run_v2(
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
    if config.algorithm_version != "v2_ssl_mil":
        raise ValueError("test_one_run_v2 requires algorithm_version=v2_ssl_mil")
    if {record.machine_type for record in records} != {config.machine_type}:
        raise ValueError("test manifest machine_type does not match the trained model")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BearingDiagnosisModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.normal_center.copy_(checkpoint["normal_center"].to(device))
    model.eval()
    threshold = float(checkpoint["evaluation_threshold"])
    preprocess, scaler = load_preprocess_state(run_dir)
    loader = DataLoader(
        BearingDataset(records, preprocess, scaler),
        batch_sampler=TimeGroupBatchSampler(records),
        collate_fn=collate_same_length,
    )
    print(f"[test-v2-setup] model={config.model_name} samples={len(records)} threshold={threshold:.6f} device={device}", flush=True)
    predictions, _ = evaluate(model, loader, device, config.mechanism_aux_weight, progress_every=50)
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
            "sample_group_id": record.sample_group_id or record.sample_id,
            "dataset_split": record.dataset_split,
        })
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[(str(row["object_id"]), str(row["sample_group_id"]))].append(row)
    bag_rows: list[dict[str, Any]] = []
    for (object_id, group_id), rows in sorted(grouped.items()):
        count = max(1, math.ceil(len(rows) * config.mil_top_fraction))
        logits = sorted((float(row["abnormal_logit"]) for row in rows), reverse=True)[:count]
        probability = float(1.0 / (1.0 + np.exp(-np.mean(logits))))
        bag_rows.append({"object_id": object_id, "sample_group_id": group_id,
                         "sample_count": len(rows), "target": int(rows[0]["target"]),
                         "abnormal_probability": probability})
    overall = _probability_summary(enriched, threshold)
    metrics: dict[str, Any] = {
        "model_name": config.model_name,
        "algorithm_version": config.algorithm_version,
        "machine_type": config.machine_type,
        "dataset_split": records[0].dataset_split,
        "evaluation_threshold_source": "frozen_development_validation_time_group_topk",
        "overall_sample_level_diagnostic_only": overall,
        "bag_level": _probability_summary(bag_rows, threshold),
        "bag_level_by_object_id": _group_summaries(bag_rows, "object_id", threshold),
        "by_sensor_position_sample_level": _group_summaries(enriched, "sensor_position", threshold),
        "by_rpm_bin_sample_level": _group_summaries(enriched, "rpm_bin", threshold),
        "by_range_position_sample_level": _group_summaries(
            [row for row in enriched if int(row["target"]) == 1], "range_position", threshold
        ),
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; run: pip install -r requirements.txt") from exc
    pq.write_table(pa.Table.from_pylist(enriched), output_dir / "predictions_test.parquet")
    pq.write_table(pa.Table.from_pylist(bag_rows), output_dir / "predictions_test_bags.parquet")
    (output_dir / "metrics_test.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[test-v2-done] samples={len(records)} bags={len(bag_rows)} output_dir={output_dir}", flush=True)
    return {"output_dir": str(output_dir), "metrics": metrics}
