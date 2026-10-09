from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("runs")
MODELS = {
    "mechanism_only": ROOT / "dfig_diagnosis_v2_ablation_mechanism_only_seed2026" / "test_nh12",
    "spectrum_only": ROOT / "dfig_diagnosis_v2_ablation_spectrum_only_seed2026" / "test_nh12",
    "gated": ROOT / "dfig_diagnosis_v2_ablation_gated_seed2026" / "test_nh12",
}
OUTPUT = ROOT / "dfig_ablation_nh12_analysis_20260916" / "step1_range_temporal_metrics.json"


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (("p50", .5), ("p90", .9), ("p95", .95))}


def persistent_metrics(rows: list[dict[str, object]], threshold: float, required: int = 3) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: datetime.fromisoformat(str(row["acquisition_time"])))
    flags = np.asarray([float(row["abnormal_probability"]) >= threshold for row in ordered], dtype=bool)
    persistent = np.zeros(flags.size, dtype=bool)
    first_index = None
    run_start = 0
    while run_start < flags.size:
        if not flags[run_start]:
            run_start += 1
            continue
        run_end = run_start + 1
        while run_end < flags.size and flags[run_end]:
            run_end += 1
        if run_end - run_start >= required:
            persistent[run_start:run_end] = True
            if first_index is None:
                first_index = run_start + required - 1
        run_start = run_end
    first = ordered[first_index] if first_index is not None else None
    return {
        "required_consecutive_records": required,
        "persistent_coverage": float(np.mean(persistent)) if flags.size else 0.0,
        "persistent_record_count": int(np.sum(persistent)),
        "first_persistent_alarm_time": None if first is None else first["acquisition_time"],
        "first_persistent_alarm_stage": None if first is None else first["range_position"],
    }


def main() -> None:
    result: dict[str, object] = {
        "definition": {
            "top_k": "highest probability records within the NH12 abnormal range",
            "persistent_alarm": "at least 3 consecutive acquisition records at or above each frozen model threshold",
            "labels": "range-level weak labels; low-score records are not assumed individually faulty",
        },
        "models": {},
    }
    for model, directory in MODELS.items():
        rows = pq.read_table(directory / "predictions_test.parquet").to_pylist()
        metrics = json.loads((directory / "metrics_test.json").read_text(encoding="utf-8"))
        threshold = float(metrics["overall"]["threshold"])
        abnormal = [row for row in rows if int(row["target"]) == 1]
        normal = [row for row in rows if int(row["target"]) == 0]
        values = np.asarray([float(row["abnormal_probability"]) for row in abnormal])
        top = {}
        for fraction in (.10, .20):
            count = max(1, int(np.ceil(values.size * fraction)))
            selected = np.sort(values)[-count:]
            top[f"top_{int(fraction * 100)}pct"] = {
                "count": count, "mean_probability": float(np.mean(selected)),
                "minimum_probability": float(np.min(selected)),
                "threshold_hit_rate": float(np.mean(selected >= threshold)),
            }
        stages = {}
        for stage in ("early", "middle", "late"):
            stage_values = np.asarray([float(row["abnormal_probability"]) for row in abnormal if row["range_position"] == stage])
            stages[stage] = {
                "count": int(stage_values.size), **quantiles(stage_values),
                "threshold_hit_rate": float(np.mean(stage_values >= threshold)),
            }
        result["models"][model] = {
            "threshold": threshold,
            "abnormal_range": {"count": len(abnormal), **quantiles(values), "threshold_hit_rate": float(np.mean(values >= threshold)), **top},
            "normal_range": {"count": len(normal), "false_positive_rate": float(np.mean([float(row["abnormal_probability"]) >= threshold for row in normal]))},
            "by_stage": stages,
            "abnormal_persistence": persistent_metrics(abnormal, threshold),
            "normal_persistence": persistent_metrics(normal, threshold),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
