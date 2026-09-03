from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .admission import write_snapshot
from .artifacts import save_preprocess_state, write_model_card, write_prediction_parquet
from .config import ModelConfig
from .dataset import (
    BalancedGroupBatchSampler,
    BearingDataset,
    LengthBatchSampler,
    MechanismScaler,
    collect_raw_mechanism_features,
)
from .evaluation import binary_metrics, select_f1_threshold
from .model import BearingDiagnosisModel, model_metadata, weak_supervision_loss
from .preprocessing import FrequencyGrid, PreprocessState, fit_amplitude_p995, load_waveform
from .schemas import SampleRecord, write_jsonl
from .splitting import assert_no_group_leakage


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_device(device_name: str | None = None) -> torch.device:
    """Resolve the requested training device without silently using the CPU."""
    requested = device_name or "cuda"
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but CUDA is unavailable. Install the CUDA "
            "PyTorch build from requirements.txt and verify the NVIDIA driver with "
            "nvidia-smi. CPU fallback is intentionally disabled."
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; detected {torch.cuda.device_count()} device(s)"
        )
    return torch.device("cuda", index)


def checkpoint_improved(
    pr_auc: float,
    validation_loss: float,
    best_pr_auc: float,
    best_validation_loss: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """Prefer PR-AUC, then use validation loss to break saturated ties."""
    if pr_auc > best_pr_auc + atol:
        return True
    return abs(pr_auc - best_pr_auc) <= atol and validation_loss < best_validation_loss


def update_convergence_count(
    validation_loss: float,
    threshold: float,
    current_count: int,
) -> int:
    return current_count + 1 if validation_loss <= threshold else 0


def collate_same_length(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = {int(item["waveform"].shape[-1]) for item in batch}  # type: ignore[index,union-attr]
    if len(lengths) != 1:
        raise ValueError("a batch must contain waveforms of the same length; use length-bucketed sampling")
    result: dict[str, object] = {"sample_id": [item["sample_id"] for item in batch]}
    for key in (
        "waveform", "spectrum", "rpm_normalized", "mechanism_features",
        "mechanism_valid_mask", "q_global", "target",
    ):
        result[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    return result


def _model_inputs(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "waveform", "spectrum", "rpm_normalized", "mechanism_features",
        "mechanism_valid_mask", "q_global",
    )
    return {key: batch[key].to(device) for key in keys}  # type: ignore[union-attr]


@torch.no_grad()
def evaluate(
    model: BearingDiagnosisModel,
    loader: DataLoader,
    device: torch.device,
    mechanism_aux_weight: float,
    progress_every: int | None = None,
) -> tuple[list[dict[str, float | str | int]], float]:
    model.eval()
    predictions: list[dict[str, float | str | int]] = []
    total_loss = 0.0
    batches = 0
    processed = 0
    total_samples = len(loader.dataset)
    next_progress = progress_every
    for batch in loader:
        output = model(**_model_inputs(batch, device))
        loss, _ = weak_supervision_loss(
            output, batch["target"].to(device), mechanism_aux_weight
        )
        total_loss += float(loss.detach())
        batches += 1
        processed += len(batch["sample_id"])
        probability = torch.sigmoid(output["abnormal_logit"]).cpu().numpy()
        auxiliary = torch.sigmoid(output["mechanism_aux_logit"]).cpu().numpy()
        targets = batch["target"].numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            predictions.append(
                {
                    "sample_id": str(sample_id),
                    "target": int(targets[index]),
                    "abnormal_probability": float(probability[index]),
                    "mechanism_aux_probability": float(auxiliary[index]),
                    "q_global": float(output["q_global"][index].cpu()),
                    "g_global": float(output["g_global"][index].cpu()),
                    "spectrum_weight": float(output["spectrum_weight"][index].cpu()),
                    "mechanism_weight": float(output["mechanism_weight"][index].cpu()),
                    "spectrum_expert_probability": float(
                        torch.sigmoid(output["spectrum_expert_logit"])[index].cpu()
                    ),
                    "mechanism_expert_probability": float(
                        torch.sigmoid(output["mechanism_expert_logit"])[index].cpu()
                    ),
                    "mechanism_to_spectrum_norm_ratio": float(
                        output["mechanism_to_spectrum_norm_ratio"][index].cpu()
                    ),
                }
            )
        if next_progress is not None and processed >= next_progress:
            print(f"[test] processed={processed}/{total_samples}", flush=True)
            while next_progress <= processed:
                next_progress += progress_every  # type: ignore[operator]
    return predictions, total_loss / max(batches, 1)


def predict(
    model: BearingDiagnosisModel, loader: DataLoader, device: torch.device
) -> list[dict[str, float | str | int]]:
    predictions, _ = evaluate(model, loader, device, mechanism_aux_weight=0.0)
    return predictions


def fit_preprocessing(train_records: list[SampleRecord], config: ModelConfig) -> tuple[PreprocessState, MechanismScaler]:
    normal = [record for record in train_records if not record.is_observed_scope_abnormal]
    if not normal:
        raise ValueError("normal training samples are required to fit preprocessing")
    signals = [load_waveform(record.waveform_path) for record in normal]
    p995 = fit_amplitude_p995(signals)
    max_theoretical = max(
        record.rpm / 60.0 * max(record.component_orders.values()) * 5.0
        for record in train_records
    )
    grid = FrequencyGrid.fit(
        [(record.sampling_rate_hz, record.waveform_length) for record in normal],
        config.business_f_max_hz,
        max_theoretical,
    )
    state = PreprocessState(
        p995,
        grid,
        config.rpm_min,
        config.rpm_max,
        config.spectrum_representation,
        config.order_suppression_harmonics,
        config.order_suppression_half_width,
        config.order_suppression_floor,
    )
    features, masks = collect_raw_mechanism_features(train_records, state)
    return state, MechanismScaler.fit(features, masks)


def train_one_run(
    config: ModelConfig,
    train_records: list[SampleRecord],
    validation_records: list[SampleRecord],
    run_dir: str | Path,
    seed: int = 2026,
    device_name: str | None = None,
) -> dict[str, object]:
    if not train_records or not validation_records:
        raise ValueError("non-empty train and validation records are required")
    if {r.machine_type for r in train_records + validation_records} != {config.machine_type}:
        raise ValueError("manifest machine_type does not match model configuration")
    set_seed(seed)
    device = resolve_training_device(device_name)
    device_description = (
        f"{device} ({torch.cuda.get_device_name(device)})"
        if device.type == "cuda"
        else str(device)
    )
    print(
        f"[setup] model={config.model_name} experiment={config.experiment} seed={seed} "
        f"device={device_description} train={len(train_records)} validation={len(validation_records)}",
        flush=True,
    )
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    assert_no_group_leakage(train_records + validation_records)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required; run: pip install -r requirements.txt") from exc
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    write_jsonl(output_dir / "split_manifest.jsonl", [record.to_dict() for record in train_records + validation_records])
    snapshot_stats = {
        "sample_count": len(train_records) + len(validation_records),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "independent_fault_event_count": len(
            {r.fault_event_id for r in train_records + validation_records if r.fault_event_id}
        ),
    }
    print("[snapshot] freezing training/validation source files...", flush=True)
    write_snapshot(output_dir / "data_snapshot.json", train_records + validation_records, snapshot_stats)
    write_jsonl(output_dir / "rejected_samples.jsonl", [])
    print("[preprocess] fitting amplitude scale, frequency grid and mechanism scaler...", flush=True)
    preprocess, scaler = fit_preprocessing(train_records, config)
    save_preprocess_state(output_dir, preprocess, scaler)
    print(
        f"[preprocess] frequency_bins={preprocess.frequency_grid.axis_hz.size} "
        f"f_max_hz={preprocess.frequency_grid.f_max_hz:g} "
        f"delta_f_hz={preprocess.frequency_grid.delta_f_hz:g}",
        flush=True,
    )
    train_dataset = BearingDataset(train_records, preprocess, scaler)
    validation_dataset = BearingDataset(validation_records, preprocess, scaler)
    train_sampler = BalancedGroupBatchSampler(train_records, config.batch_size, seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_same_length)
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=LengthBatchSampler(validation_records, config.batch_size),
        collate_fn=collate_same_length,
    )
    model = BearingDiagnosisModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_pr_auc = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    patience = 0
    convergence_count = 0
    stop_reason = "max_epochs"
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**_model_inputs(batch, device))
            loss, _ = weak_supervision_loss(
                outputs, batch["target"].to(device), config.mechanism_aux_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation_predictions, validation_loss = evaluate(
            model, validation_loader, device, config.mechanism_aux_weight
        )
        targets = [int(row["target"]) for row in validation_predictions]
        probabilities = [float(row["abnormal_probability"]) for row in validation_predictions]
        threshold = select_f1_threshold(targets, probabilities)
        metrics = binary_metrics(targets, probabilities, threshold)
        current = float(metrics["pr_auc_range_label"])
        train_loss = total_loss / max(batches, 1)
        convergence_count = update_convergence_count(
            validation_loss,
            config.convergence_val_loss,
            convergence_count,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_pr_auc": current,
            "validation_f1": float(metrics["f1"]),
            "threshold": threshold,
            "convergence_count": convergence_count,
        })
        if checkpoint_improved(
            current,
            validation_loss,
            best_pr_auc,
            best_validation_loss,
        ):
            best_pr_auc = current
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "config": asdict(config),
                    "evaluation_threshold": threshold,
                    "best_epoch": best_epoch,
                    "best_validation_pr_auc": best_pr_auc,
                    "best_validation_loss": best_validation_loss,
                    "metadata": model_metadata(model),
                },
                output_dir / "model_best_during_training.pt",
            )
            patience = 0
        else:
            patience += 1
        elapsed = time.perf_counter() - epoch_started
        print(
            f"[epoch {epoch:03d}/{config.max_epochs}] train_loss={train_loss:.5f} "
            f"val_loss={validation_loss:.5f} pr_auc={current:.4f} "
            f"f1={float(metrics['f1']):.4f} patience={patience}/{config.early_stopping_patience} "
            f"low_loss={convergence_count}/{config.convergence_epochs} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if convergence_count >= config.convergence_epochs:
            stop_reason = "converged_validation_loss"
            print(
                f"[converged] val_loss <= {config.convergence_val_loss:g} for "
                f"{config.convergence_epochs} consecutive epochs",
                flush=True,
            )
            break
        if patience >= config.early_stopping_patience:
            stop_reason = "early_stopping_patience"
            print(
                f"[early-stop] no PR-AUC or tie-break validation-loss improvement "
                f"for {patience} epochs",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    validation_predictions = predict(model, validation_loader, device)
    targets = [int(row["target"]) for row in validation_predictions]
    probabilities = [float(row["abnormal_probability"]) for row in validation_predictions]
    threshold = select_f1_threshold(targets, probabilities)
    metrics = binary_metrics(targets, probabilities, threshold)
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "evaluation_threshold": threshold,
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "best_validation_loss": best_validation_loss,
        "stop_reason": stop_reason,
        "metadata": model_metadata(model),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    save_preprocess_state(output_dir, preprocess, scaler)
    (output_dir / "metrics_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_prediction_parquet(output_dir, validation_predictions, "validation")
    write_model_card(output_dir, config, threshold, metrics)
    print(
        f"[done] best_epoch={best_epoch} pr_auc={float(metrics['pr_auc_range_label']):.4f} "
        f"f1={float(metrics['f1']):.4f} threshold={threshold:.6f} run_dir={output_dir}",
        flush=True,
    )
    return {"run_dir": str(output_dir), "threshold": threshold, "metrics": metrics}
