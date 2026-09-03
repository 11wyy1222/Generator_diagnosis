from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .dataset import MechanismScaler
from .preprocessing import FrequencyGrid, PreprocessState


def save_preprocess_state(run_dir: str | Path, state: PreprocessState, scaler: MechanismScaler) -> None:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "frequency_grid.npy", state.frequency_grid.axis_hz)
    np.savez(output / "mechanism_scaler.npz", center=scaler.center, scale=scaler.scale)
    metadata = {
        "amplitude_p995": state.amplitude_p995,
        "rpm_min": state.rpm_min,
        "rpm_max": state.rpm_max,
        "spectrum_representation": state.spectrum_representation,
        "order_suppression_harmonics": list(state.order_suppression_harmonics),
        "order_suppression_half_width": state.order_suppression_half_width,
        "order_suppression_floor": state.order_suppression_floor,
        "frequency_grid": {
            "f_max_hz": state.frequency_grid.f_max_hz,
            "delta_f_hz": state.frequency_grid.delta_f_hz,
            "frequency_bin_count": int(state.frequency_grid.axis_hz.size),
        },
    }
    if state.spectrum_representation == "shaft_order_v2":
        metadata["order_grid"] = {
            "max_order": float(state.order_axis[-1]),
            "delta_order": float(state.order_axis[1] - state.order_axis[0]),
            "order_bin_count": int(state.order_axis.size),
            "uses_theoretical_fault_orders": False,
        }
    (output / "preprocess.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "frequency_grid_metadata.json").write_text(
        json.dumps(metadata["frequency_grid"], indent=2), encoding="utf-8"
    )


def load_preprocess_state(run_dir: str | Path) -> tuple[PreprocessState, MechanismScaler]:
    source = Path(run_dir)
    metadata = json.loads((source / "preprocess.json").read_text(encoding="utf-8"))
    axis = np.load(source / "frequency_grid.npy", allow_pickle=False)
    scaler_data = np.load(source / "mechanism_scaler.npz", allow_pickle=False)
    grid = FrequencyGrid(axis, metadata["frequency_grid"]["f_max_hz"], metadata["frequency_grid"]["delta_f_hz"])
    return (
        PreprocessState(
            metadata["amplitude_p995"],
            grid,
            metadata["rpm_min"],
            metadata["rpm_max"],
            metadata.get("spectrum_representation", "frequency_hz_v1"),
            tuple(metadata.get("order_suppression_harmonics", (1.0, 2.0, 3.0))),
            metadata.get("order_suppression_half_width", 0.08),
            metadata.get("order_suppression_floor", 0.25),
        ),
        MechanismScaler(scaler_data["center"], scaler_data["scale"]),
    )


def write_model_card(run_dir: str | Path, config: ModelConfig, threshold: float, metrics: dict[str, object]) -> None:
    content = f"""# Model card: {config.model_name}

- Machine type: `{config.machine_type}`
- Experiment: `{config.experiment}`
- Spectrum representation: `{config.spectrum_representation}`
- Gated fusion: `{config.gated_fusion}`
- Configuration SHA-256: `{config.config_hash}`
- Evaluation threshold: `{threshold:.8g}` (chosen for maximum validation F1)
- Component classification heads: disabled

## Intended output

The external result contains only `sample_id`, `abnormal_probability`, and a null
`component_probabilities` field. The score is learned from time-range propagated weak
labels. It is not a calibrated probability against waveform-level expert truth and
must not be presented as an automatic business conclusion.

## Validation scope

Metrics are computed against range labels using leakage-safe time-block splits. They
do not establish waveform-level clinical/engineering sensitivity or independent-event
generalization. DFIG F14 is an abnormal-only, cross-project and cross-sensor-position
transfer check; it cannot measure specificity.

## Validation metrics

```json
{json.dumps(metrics, ensure_ascii=False, indent=2)}
```
"""
    Path(run_dir, "model_card.md").write_text(content, encoding="utf-8")


def write_prediction_parquet(run_dir: str | Path, predictions: list[dict[str, object]], name: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required; run: pip install -r requirements.txt") from exc
    output = Path(run_dir)
    external_columns = ("sample_id", "target", "abnormal_probability")
    gate_columns = (
        "sample_id", "mechanism_aux_probability", "q_global", "g_global",
        "spectrum_weight", "mechanism_weight", "spectrum_expert_probability",
        "mechanism_expert_probability", "mechanism_to_spectrum_norm_ratio",
    )
    external = [{key: row[key] for key in external_columns} for row in predictions]
    gates = [{key: row[key] for key in gate_columns} for row in predictions]
    pq.write_table(pa.Table.from_pylist(external), output / f"predictions_{name}.parquet")
    pq.write_table(pa.Table.from_pylist(gates), output / "gate_monitoring.parquet")
