from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bearing_diagnosis.evaluation import binary_metrics


ROOT = Path("runs")
MODELS = {
    "mechanism_only": ROOT / "dfig_diagnosis_v2_ablation_mechanism_only_seed2026" / "test_nh12",
    "spectrum_only": ROOT / "dfig_diagnosis_v2_ablation_spectrum_only_seed2026" / "test_nh12",
    "gated": ROOT / "dfig_diagnosis_v2_ablation_gated_seed2026" / "test_nh12",
}
OUTPUT = ROOT / "dfig_ablation_nh12_analysis_20260916"


def empirical_probability(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / (ordered.size + 1.0)


def main() -> None:
    summary: dict[str, object] = {
        "protocol": {
            "normal_calibration": "earliest 60% of NH12 pre-fault normal records",
            "normal_evaluation": "latest 40% of NH12 pre-fault normal records",
            "abnormal_evaluation": "all NH12 abnormal-range records",
            "weight_updates": False,
            "threshold": "P95 empirical percentile of calibration-normal scores",
        },
        "models": {},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model, directory in MODELS.items():
        rows = pq.read_table(directory / "predictions_test.parquet").to_pylist()
        normal = sorted(
            [row for row in rows if int(row["target"]) == 0],
            key=lambda row: datetime.fromisoformat(str(row["acquisition_time"])),
        )
        cut = int(np.floor(len(normal) * .60))
        calibration_ids = {str(row["sample_id"]) for row in normal[:cut]}
        heldout_ids = {str(row["sample_id"]) for row in normal[cut:]}
        calibration_scores = np.asarray([float(row["abnormal_probability"]) for row in normal[:cut]])
        calibration_probabilities = empirical_probability(calibration_scores, calibration_scores)
        threshold = float(np.quantile(calibration_probabilities, .95))
        output_rows = []
        for row in rows:
            probability = float(empirical_probability(calibration_scores, np.asarray([float(row["abnormal_probability"])]))[0])
            sample_id = str(row["sample_id"])
            role = "normal_calibration" if sample_id in calibration_ids else "normal_heldout" if sample_id in heldout_ids else "abnormal_test"
            output_rows.append({
                **row, "original_abnormal_probability": float(row["abnormal_probability"]),
                "calibrated_probability": probability, "calibration_role": role,
                "calibrated_predicted_abnormal": probability >= threshold,
            })
        evaluation = [row for row in output_rows if row["calibration_role"] != "normal_calibration"]
        targets = [int(row["target"]) for row in evaluation]
        probabilities = [float(row["calibrated_probability"]) for row in evaluation]
        metrics = binary_metrics(targets, probabilities, threshold)
        calibration_fpr = float(np.mean(calibration_probabilities >= threshold))
        summary["models"][model] = {
            "calibration_normal_count": cut,
            "heldout_normal_count": len(normal) - cut,
            "abnormal_test_count": sum(int(row["target"]) == 1 for row in rows),
            "threshold": threshold,
            "calibration_false_positive_rate": calibration_fpr,
            **metrics,
        }
        pq.write_table(pa.Table.from_pylist(output_rows), OUTPUT / f"step2_{model}_calibrated_predictions.parquet")
    (OUTPUT / "step2_target_normal_calibration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
