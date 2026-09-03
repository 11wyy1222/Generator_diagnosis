from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .artifacts import write_model_card
from .config import ModelConfig
from .evaluation import binary_metrics, select_f1_threshold


def calibrate_threshold(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"calibrated output directory already exists: {output}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required; run: pip install -r requirements.txt") from exc

    prediction_path = source / "predictions_validation.parquet"
    rows = pq.read_table(prediction_path).to_pylist()
    targets = [int(row["target"]) for row in rows]
    probabilities = [float(row["abnormal_probability"]) for row in rows]
    new_threshold = select_f1_threshold(targets, probabilities)
    metrics = binary_metrics(targets, probabilities, new_threshold)

    checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    old_threshold = float(checkpoint["evaluation_threshold"])
    raw_config = dict(checkpoint["config"])
    raw_config["random_seeds"] = tuple(raw_config["random_seeds"])
    config = ModelConfig(**raw_config)

    # Preserve the trained weights and all training artifacts, while excluding
    # prior test result directories so the calibrated version can be tested
    # independently without overwriting the original report.
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("test_*"))
    checkpoint["evaluation_threshold"] = new_threshold
    torch.save(checkpoint, output / "model.pt")
    (output / "metrics_validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_model_card(output, config, new_threshold, metrics)
    calibration = {
        "method": "midpoint_between_adjacent_validation_scores_maximizing_f1",
        "source": "predictions_validation.parquet",
        "network_weights_changed": False,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "validation_sample_count": len(rows),
        "validation_metrics": metrics,
    }
    (output / "threshold_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[calibrate] old_threshold={old_threshold:.6f} "
        f"new_threshold={new_threshold:.6f} output_dir={output}",
        flush=True,
    )
    return {"output_dir": str(output), **calibration}
