from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .artifacts import load_preprocess_state
from .config import ModelConfig
from .mechanism import extract_mechanism_features
from .model import BearingDiagnosisModel
from .preprocessing import load_waveform, native_spectra, validate_waveform


@dataclass(frozen=True)
class InferenceInput:
    sample_id: str
    waveform_path: str
    sampling_rate_hz: float
    rpm: float
    component_orders: dict[str, float]


class BearingInference:
    def __init__(self, run_dir: str | Path, device_name: str = "cpu") -> None:
        self.run_dir = Path(run_dir)
        self.device = torch.device(device_name)
        checkpoint = torch.load(self.run_dir / "model.pt", map_location=self.device, weights_only=False)
        config_values = dict(checkpoint["config"])
        config_values["random_seeds"] = tuple(config_values["random_seeds"])
        self.config = ModelConfig(**config_values)
        self.model = BearingDiagnosisModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(checkpoint["evaluation_threshold"])
        self.preprocess, self.scaler = load_preprocess_state(self.run_dir)

    @torch.no_grad()
    def predict(self, item: InferenceInput) -> tuple[dict[str, object], dict[str, object]]:
        signal = load_waveform(item.waveform_path)
        reasons = validate_waveform(signal)
        if reasons:
            raise ValueError(f"rejected waveform {item.sample_id}: {', '.join(reasons)}")
        frequency, ordinary, envelope = native_spectra(signal, item.sampling_rate_hz)
        spectrum = self.preprocess.transform_spectrum(frequency, ordinary, envelope, item.rpm)
        evidence = extract_mechanism_features(
            frequency, ordinary, envelope, item.rpm, item.component_orders,
            item.sampling_rate_hz, signal.size,
        )
        features = self.scaler.transform(evidence.features, evidence.valid_mask)
        outputs = self.model(
            waveform=torch.from_numpy(self.preprocess.normalize_time(signal)[None, None, :]).to(self.device),
            spectrum=torch.from_numpy(spectrum[None, ...]).to(self.device),
            rpm_normalized=torch.tensor([self.preprocess.normalize_rpm(item.rpm)], dtype=torch.float32, device=self.device),
            mechanism_features=torch.from_numpy(features[None, ...]).to(self.device),
            mechanism_valid_mask=torch.from_numpy(evidence.valid_mask[None, ...]).to(self.device),
            q_global=torch.tensor([evidence.q_global], dtype=torch.float32, device=self.device),
        )
        probability = float(torch.sigmoid(outputs["abnormal_logit"])[0].cpu())
        external = {
            "sample_id": item.sample_id,
            "abnormal_probability": probability,
            "component_probabilities": None,
        }
        internal = {
            "model_name": self.config.model_name,
            "config_hash": self.config.config_hash,
            "sampling_rate_hz": item.sampling_rate_hz,
            "waveform_length": int(signal.size),
            "rpm": item.rpm,
            "fft_resolution_hz": item.sampling_rate_hz / signal.size,
            "q_global": evidence.q_global,
            "g_global": float(outputs["g_global"][0].cpu()),
            "spectrum_weight": float(outputs["spectrum_weight"][0].cpu()),
            "mechanism_weight": float(outputs["mechanism_weight"][0].cpu()),
            "spectrum_expert_probability": float(
                torch.sigmoid(outputs["spectrum_expert_logit"])[0].cpu()
            ),
            "mechanism_expert_probability": float(
                torch.sigmoid(outputs["mechanism_expert_logit"])[0].cpu()
            ),
            "mechanism_aux_probability": float(torch.sigmoid(outputs["mechanism_aux_logit"])[0].cpu()),
            "mechanism_to_spectrum_norm_ratio": float(outputs["mechanism_to_spectrum_norm_ratio"][0].cpu()),
            "degraded_to_spectrum_only": evidence.q_global == 0.0,
            "theoretical_frequencies_hz": evidence.theoretical_frequencies_hz,
        }
        return external, internal
