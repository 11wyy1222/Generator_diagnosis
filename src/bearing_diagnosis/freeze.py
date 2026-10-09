from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ARTIFACTS = (
    "model.pt",
    "run_config.yaml",
    "preprocess.json",
    "frequency_grid.npy",
    "frequency_grid_metadata.json",
    "mechanism_scaler.npz",
)

PIPELINE_SOURCES = (
    "src/bearing_diagnosis/artifacts.py",
    "src/bearing_diagnosis/config.py",
    "src/bearing_diagnosis/dataset.py",
    "src/bearing_diagnosis/mechanism.py",
    "src/bearing_diagnosis/model.py",
    "src/bearing_diagnosis/preprocessing.py",
    "src/bearing_diagnosis/schemas.py",
    "src/bearing_diagnosis/testing.py",
    "src/bearing_diagnosis/training.py",
)


def _digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": hasher.hexdigest()}


def create_freeze_manifest(
    run_dir: str | Path, output_path: str | Path, source_root: str | Path = ".",
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from exc
    run = Path(run_dir).resolve()
    source = Path(source_root).resolve()
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"freeze manifest already exists: {output}")
    artifacts = [run / name for name in RUN_ARTIFACTS]
    sources = [source / name for name in PIPELINE_SOURCES]
    missing = [str(path) for path in artifacts + sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("freeze inputs missing: " + ", ".join(missing))
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    algorithm_version = str(config.get("algorithm_version", "v1_sample_bce"))
    if algorithm_version != "v1_sample_bce":
        raise ValueError(f"only v1_sample_bce can be frozen for this protocol: {algorithm_version}")
    manifest = {
        "freeze_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "target-normal threshold calibration without weight or preprocessing updates",
        "run_dir": str(run),
        "model_name": config["model_name"],
        "algorithm_version": algorithm_version,
        "experiment": config["experiment"],
        "network_weights_frozen": True,
        "preprocessing_frozen": True,
        "feature_computation_frozen": True,
        "run_artifacts": [_digest(path) for path in artifacts],
        "pipeline_sources": [_digest(path) for path in sources],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_freeze_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["run_artifacts"] + manifest["pipeline_sources"]:
        file_path = Path(item["path"])
        if not file_path.is_file():
            mismatches.append(f"missing:{file_path}")
            continue
        actual = _digest(file_path)
        if actual["sha256"] != item["sha256"] or actual["size"] != item["size"]:
            mismatches.append(f"changed:{file_path}")
    if mismatches:
        raise ValueError("frozen model/pipeline verification failed: " + ", ".join(mismatches))
    return manifest
