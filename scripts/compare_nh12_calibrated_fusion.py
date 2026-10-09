from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bearing_diagnosis.evaluation import binary_metrics


ROOT = Path("runs/dfig_ablation_nh12_analysis_20260916")
MODELS = ("mechanism_only", "spectrum_only", "gated")


def load_rows(model: str) -> dict[str, dict[str, object]]:
    path = ROOT / f"step2_{model}_calibrated_predictions.parquet"
    return {str(row["sample_id"]): row for row in pq.read_table(path).to_pylist()}


def main() -> None:
    step2 = json.loads((ROOT / "step2_target_normal_calibration.json").read_text(encoding="utf-8"))
    source = {model: load_rows(model) for model in MODELS}
    common = sorted(set.intersection(*(set(rows) for rows in source.values())))
    fused_rows = []
    for sample_id in common:
        mechanism = source["mechanism_only"][sample_id]
        spectrum = source["spectrum_only"][sample_id]
        probability = .60 * float(mechanism["calibrated_probability"]) + .40 * float(spectrum["calibrated_probability"])
        fused_rows.append({
            "sample_id": sample_id,
            "target": int(mechanism["target"]),
            "acquisition_time": mechanism["acquisition_time"],
            "range_position": mechanism["range_position"],
            "calibration_role": mechanism["calibration_role"],
            "mechanism_calibrated_probability": float(mechanism["calibrated_probability"]),
            "spectrum_calibrated_probability": float(spectrum["calibrated_probability"]),
            "fused_probability": probability,
        })
    calibration = np.asarray([float(row["fused_probability"]) for row in fused_rows if row["calibration_role"] == "normal_calibration"])
    threshold = float(np.quantile(calibration, .95))
    for row in fused_rows:
        row["predicted_abnormal"] = float(row["fused_probability"]) >= threshold
    evaluation = [row for row in fused_rows if row["calibration_role"] != "normal_calibration"]
    fusion_metrics = binary_metrics(
        [int(row["target"]) for row in evaluation],
        [float(row["fused_probability"]) for row in evaluation],
        threshold,
    )
    fusion_metrics.update({
        "threshold": threshold,
        "calibration_false_positive_rate": float(np.mean(calibration >= threshold)),
        "weights": {"mechanism_only": .60, "spectrum_only": .40},
    })
    comparison = {
        "protocol": {
            "weights_frozen_before_nh12_comparison": True,
            "fusion": "0.60 * calibrated mechanism percentile + 0.40 * calibrated spectrum percentile",
            "fusion_threshold": "P95 of fused earliest-60%-normal calibration scores",
            "evaluation": "latest-40%-normal plus all abnormal-range records",
        },
        "models": {model: step2["models"][model] for model in MODELS},
        "fixed_fusion": fusion_metrics,
    }
    pq.write_table(pa.Table.from_pylist(fused_rows), ROOT / "step3_fixed_fusion_predictions.parquet")
    (ROOT / "step3_calibrated_model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
