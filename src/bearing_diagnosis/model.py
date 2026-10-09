from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .config import ModelConfig
from .mechanism import FEATURE_COUNT


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class TimeEncoder(nn.Module):
    def __init__(self, pool_segments: int = 8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.GroupNorm(8, 32), nn.GELU())
        self.multiscale = nn.ModuleList([nn.Conv1d(32, 32, k, padding=k // 2) for k in (7, 15, 31)])
        self.merge = nn.Sequential(nn.Conv1d(96, 32, 1), nn.GroupNorm(8, 32), nn.GELU())
        self.blocks = nn.Sequential(ConvBlock(32, 32, 7), ConvBlock(32, 64, 15), ConvBlock(64, 128, 31))
        self.pool = nn.AdaptiveAvgPool1d(pool_segments)
        self.projection = nn.Linear(128 * pool_segments, 128)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.stem(waveform)
        x = self.merge(torch.cat([branch(x) for branch in self.multiscale], dim=1))
        return self.projection(self.pool(self.blocks(x)).flatten(1))


class SpectrumEncoder(nn.Module):
    def __init__(self, pool_segments: int = 8):
        super().__init__()
        self.blocks = nn.Sequential(ConvBlock(2, 32, 9), ConvBlock(32, 64, 7), ConvBlock(64, 128, 5))
        self.pool = nn.AdaptiveAvgPool1d(pool_segments)
        self.projection = nn.Linear(128 * pool_segments, 128)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.blocks(spectrum)).flatten(1))


def mlp(dims: list[int], dropout: float = 0.0, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(left, right))
        is_last = index == len(dims) - 2
        if final_activation or not is_last:
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class BearingDiagnosisModel(nn.Module):
    """A/B/C/D weak-label binary classifier described by the implementation plan."""

    def __init__(self, config: ModelConfig, mechanism_feature_count: int = FEATURE_COUNT):
        super().__init__()
        self.config = config
        self.time_encoder = TimeEncoder(config.time_pool_segments)
        self.spectrum_encoder = SpectrumEncoder(config.spectrum_pool_segments)
        self.spectrum_fusion = mlp([257, 128, 64], config.dropout)
        self.shared_mechanism = mlp([mechanism_feature_count * 2, 64, 32])
        self.global_mechanism = mlp([32, 64])
        self.mechanism_aux_head = mlp([64, 32, 1], final_activation=False)
        self.spectrum_only_head = mlp([64, 64, 32, 1], config.dropout, final_activation=False)
        self.mechanism_only_head = mlp([64, 64, 32, 1], config.dropout, final_activation=False)
        self.concat_head = mlp([128, 64, 32, 1], config.dropout, final_activation=False)
        self.global_projection = nn.Linear(64, 64)
        self.global_gate = mlp([129, 64, 1], final_activation=False)
        self.layer_norm = nn.LayerNorm(64)
        self.gated_head = mlp([128, 64, 32, 1], config.dropout, final_activation=False)
        # Stored separately in v2 checkpoints so historical v1 state dicts remain strict-load compatible.
        self.register_buffer("normal_center", torch.zeros(64), persistent=False)

    def encode_spectrum(self, waveform: torch.Tensor, spectrum: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        z_raw = self.time_encoder(waveform)
        z_spectrum = self.spectrum_encoder(spectrum)
        return self.spectrum_fusion(torch.cat([z_raw, z_spectrum, rpm.reshape(-1, 1)], dim=1))

    def encode_mechanism(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        masked = features * valid_mask
        batch, components, _ = masked.shape
        encoded = self.shared_mechanism(torch.cat([masked, valid_mask], dim=-1).reshape(batch * components, -1))
        return self.global_mechanism(encoded.reshape(batch, components, -1).mean(dim=1))

    def forward(
        self,
        waveform: torch.Tensor,
        spectrum: torch.Tensor,
        rpm_normalized: torch.Tensor,
        mechanism_features: torch.Tensor,
        mechanism_valid_mask: torch.Tensor,
        q_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_spec = self.encode_spectrum(waveform, spectrum, rpm_normalized)
        z_phy = self.encode_mechanism(mechanism_features, mechanism_valid_mask)
        q = q_global.reshape(-1, 1).clamp(0.0, 1.0)
        auxiliary_logit = self.mechanism_aux_head(z_phy).squeeze(1)
        gate = torch.zeros_like(q)
        gated_phy = torch.zeros_like(z_spec)
        if self.config.experiment == "spectrum_only":
            abnormal_logit = self.spectrum_only_head(z_spec).squeeze(1)
        elif self.config.experiment == "mechanism_only":
            abnormal_logit = self.mechanism_only_head(z_phy).squeeze(1)
        elif self.config.experiment == "concat":
            abnormal_logit = self.concat_head(torch.cat([z_spec, z_phy], dim=1)).squeeze(1)
        else:
            projected = self.global_projection(z_phy)
            gate = q * torch.sigmoid(self.global_gate(torch.cat([z_spec, projected, q], dim=1)))
            gated_phy = gate * projected
            fused = self.layer_norm(z_spec + gated_phy)
            abnormal_logit = self.gated_head(torch.cat([fused, z_spec * gated_phy], dim=1)).squeeze(1)
        return {
            "abnormal_logit": abnormal_logit,
            "mechanism_aux_logit": auxiliary_logit,
            "q_global": q.squeeze(1),
            "g_global": gate.squeeze(1),
            "mechanism_to_spectrum_norm_ratio": gated_phy.norm(dim=1) / z_spec.norm(dim=1).clamp_min(1e-8),
            "deep_embedding": z_spec,
            "normal_distance": ((z_spec - self.normal_center) ** 2).mean(dim=1),
        }


def topk_mil_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    top_fraction: float = 0.20,
    auxiliary_weight: float = 0.10,
    normal_center_weight: float = 0.10,
    bag_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weak bag loss: all normal instances low; only positive-bag top-k must be high."""
    targets = targets.float().reshape(-1)
    if targets.numel() == 0 or torch.unique(targets).numel() != 1:
        raise ValueError("one MIL batch must be one non-empty, label-homogeneous time group")
    label = targets[0]
    logits = outputs["abnormal_logit"].reshape(-1)
    aux_logits = outputs["mechanism_aux_logit"].reshape(-1)
    q = outputs["q_global"].reshape(-1)
    if label.item() >= 0.5:
        k = max(1, int(torch.ceil(torch.tensor(logits.numel() * top_fraction)).item()))
        main = nn.functional.binary_cross_entropy_with_logits(logits.topk(k).values.mean(), label)
        valid_aux = aux_logits[q > 0]
        if valid_aux.numel():
            aux_k = min(k, valid_aux.numel())
            auxiliary = nn.functional.binary_cross_entropy_with_logits(
                valid_aux.topk(aux_k).values.mean(), label
            )
        else:
            auxiliary = logits.new_zeros(())
        center = logits.new_zeros(())
    else:
        main = nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).mean()
        per_aux = nn.functional.binary_cross_entropy_with_logits(aux_logits, torch.zeros_like(aux_logits))
        auxiliary = (per_aux * q).sum() / q.sum().clamp_min(torch.finfo(q.dtype).eps)
        center = outputs["normal_distance"].mean()
    total = float(bag_weight) * (
        main + float(auxiliary_weight) * auxiliary + float(normal_center_weight) * center
    )
    return total, {
        "main_loss": float(main.detach()),
        "auxiliary_loss": float(auxiliary.detach()),
        "normal_center_loss": float(center.detach()),
    }


def ssl_consistency_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    variance_weight: float = 0.10,
    covariance_weight: float = 0.01,
) -> torch.Tensor:
    """VICReg-style normal-only consistency objective for the deep embedding."""
    invariance = nn.functional.mse_loss(
        nn.functional.normalize(first, dim=1), nn.functional.normalize(second, dim=1)
    )
    first_centered = first - first.mean(0)
    second_centered = second - second.mean(0)
    std_penalty = torch.relu(1.0 - torch.sqrt(first.var(0, unbiased=False) + 1e-4)).mean()
    std_penalty = std_penalty + torch.relu(1.0 - torch.sqrt(second.var(0, unbiased=False) + 1e-4)).mean()
    if first.shape[0] > 1:
        cov_first = first_centered.T @ first_centered / (first.shape[0] - 1)
        cov_second = second_centered.T @ second_centered / (second.shape[0] - 1)
        eye = torch.eye(first.shape[1], device=first.device, dtype=first.dtype)
        covariance = ((cov_first * (1 - eye)) ** 2).mean() + ((cov_second * (1 - eye)) ** 2).mean()
    else:
        covariance = first.new_zeros(())
    return invariance + float(variance_weight) * std_penalty + float(covariance_weight) * covariance


def weak_supervision_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    auxiliary_weight: float = 0.10,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = targets.float().reshape(-1)
    weighted_training = sample_weights is not None
    weights = (
        torch.ones_like(targets)
        if sample_weights is None
        else sample_weights.to(device=targets.device, dtype=targets.dtype).reshape(-1)
    )
    if weights.shape != targets.shape or torch.any(weights < 0) or not torch.isfinite(weights).all():
        raise ValueError("sample_weights must be finite, non-negative and match targets")
    per_sample_main = nn.functional.binary_cross_entropy_with_logits(
        outputs["abnormal_logit"], targets, reduction="none"
    )
    per_sample_aux = nn.functional.binary_cross_entropy_with_logits(
        outputs["mechanism_aux_logit"], targets, reduction="none"
    )
    q = outputs["q_global"].reshape(-1)
    if weighted_training:
        main = (per_sample_main * weights).mean()
        auxiliary = (per_sample_aux * weights * q).mean()
    else:
        # Preserve the historical unweighted validation/test loss semantics.
        main = per_sample_main.mean()
        auxiliary = (per_sample_aux * q).sum() / q.sum().clamp_min(
            torch.finfo(q.dtype).eps
        )
    total = main + float(auxiliary_weight) * auxiliary
    return total, {"main_loss": float(main.detach()), "auxiliary_loss": float(auxiliary.detach())}


def model_metadata(model: BearingDiagnosisModel) -> dict[str, Any]:
    return {
        "model_name": model.config.model_name,
        "machine_type": model.config.machine_type,
        "experiment": model.config.experiment,
        "algorithm_version": model.config.algorithm_version,
        "config_hash": model.config.config_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "external_outputs": ["sample_id", "abnormal_probability", "component_probabilities"],
        "component_heads_enabled": False,
    }
