from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

Experiment = Literal["spectrum_only", "mechanism_only", "concat", "gated"]
AlgorithmVersion = Literal["v1_sample_bce", "v2_ssl_mil"]
SignalProcessingMode = Literal["native", "vibration_analysis"]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    machine_type: Literal["semi_direct", "dfig"]
    experiment: Experiment = "gated"
    # Data/split versions are intentionally separate from this algorithm version.
    algorithm_version: AlgorithmVersion = "v1_sample_bce"
    signal_processing_mode: SignalProcessingMode = "native"
    rpm_min: float = 180.0
    rpm_max: float = 350.0
    business_f_max_hz: float = 5000.0
    split_seed: int = 2026
    random_seeds: tuple[int, ...] = (2026, 2027, 2028)
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
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
    ssl_pretrain_epochs: int = 10
    ssl_noise_std_fraction: float = 0.005
    ssl_shift_fraction: float = 0.01
    ssl_variance_weight: float = 0.10
    ssl_covariance_weight: float = 0.01
    mil_top_fraction: float = 0.20
    normal_center_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.rpm_max <= self.rpm_min:
            raise ValueError("rpm_max must be greater than rpm_min")
        if self.experiment not in {"spectrum_only", "mechanism_only", "concat", "gated"}:
            raise ValueError(f"unsupported experiment: {self.experiment}")
        if self.algorithm_version not in {"v1_sample_bce", "v2_ssl_mil"}:
            raise ValueError(f"unsupported algorithm_version: {self.algorithm_version}")
        if self.signal_processing_mode not in {"native", "vibration_analysis"}:
            raise ValueError(f"unsupported signal_processing_mode: {self.signal_processing_mode}")
        if self.machine_type not in {"semi_direct", "dfig"}:
            raise ValueError(f"unsupported machine_type: {self.machine_type}")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1 or self.max_epochs < 1:
            raise ValueError(
                "batch_size, gradient_accumulation_steps and max_epochs must be positive"
            )
        if self.convergence_val_loss < 0 or self.convergence_epochs < 1:
            raise ValueError("convergence_val_loss must be non-negative and convergence_epochs positive")
        if self.ssl_pretrain_epochs < 0:
            raise ValueError("ssl_pretrain_epochs must be non-negative")
        if not 0 < self.mil_top_fraction <= 1:
            raise ValueError("mil_top_fraction must be in (0, 1]")

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
    return ModelConfig(**raw)
