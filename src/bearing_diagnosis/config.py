from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

Experiment = Literal["spectrum_only", "mechanism_only", "concat", "gated"]
SpectrumRepresentation = Literal["frequency_hz_v1", "shaft_order_v2"]
GatedFusion = Literal["spectrum_residual_v1", "competitive_v2"]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    machine_type: Literal["semi_direct", "dfig"]
    experiment: Experiment = "gated"
    rpm_min: float = 180.0
    rpm_max: float = 350.0
    business_f_max_hz: float = 5000.0
    split_seed: int = 2026
    random_seeds: tuple[int, ...] = (2026, 2027, 2028)
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 15
    convergence_val_loss: float = 0.01
    convergence_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    mechanism_aux_weight: float = 0.10
    time_pool_segments: int = 8
    spectrum_pool_segments: int = 8
    dropout: float = 0.2
    spectrum_representation: SpectrumRepresentation = "frequency_hz_v1"
    gated_fusion: GatedFusion = "spectrum_residual_v1"
    order_suppression_harmonics: tuple[float, ...] = (1.0, 2.0, 3.0)
    order_suppression_half_width: float = 0.08
    order_suppression_floor: float = 0.25

    def __post_init__(self) -> None:
        if self.rpm_max <= self.rpm_min:
            raise ValueError("rpm_max must be greater than rpm_min")
        if self.experiment not in {"spectrum_only", "mechanism_only", "concat", "gated"}:
            raise ValueError(f"unsupported experiment: {self.experiment}")
        if self.machine_type not in {"semi_direct", "dfig"}:
            raise ValueError(f"unsupported machine_type: {self.machine_type}")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.convergence_val_loss < 0 or self.convergence_epochs < 1:
            raise ValueError("convergence_val_loss must be non-negative and convergence_epochs positive")
        if self.spectrum_representation not in {"frequency_hz_v1", "shaft_order_v2"}:
            raise ValueError(f"unsupported spectrum_representation: {self.spectrum_representation}")
        if self.gated_fusion not in {"spectrum_residual_v1", "competitive_v2"}:
            raise ValueError(f"unsupported gated_fusion: {self.gated_fusion}")
        if self.gated_fusion == "competitive_v2" and self.spectrum_representation != "shaft_order_v2":
            raise ValueError("competitive_v2 requires shaft_order_v2 spectrum representation")
        if not self.order_suppression_harmonics or any(value <= 0 for value in self.order_suppression_harmonics):
            raise ValueError("order_suppression_harmonics must contain positive orders")
        if self.order_suppression_half_width <= 0:
            raise ValueError("order_suppression_half_width must be positive")
        if not 0.0 <= self.order_suppression_floor <= 1.0:
            raise ValueError("order_suppression_floor must be in [0, 1]")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> ModelConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("PyYAML is required; run: pip install -r requirements.txt") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if "random_seeds" in raw:
        raw["random_seeds"] = tuple(int(seed) for seed in raw["random_seeds"])
    if "order_suppression_harmonics" in raw:
        raw["order_suppression_harmonics"] = tuple(
            float(value) for value in raw["order_suppression_harmonics"]
        )
    return ModelConfig(**raw)
