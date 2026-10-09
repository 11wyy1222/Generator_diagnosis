from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .admission import write_snapshot
from .artifacts import save_preprocess_state, write_model_card, write_prediction_parquet
from .config import ModelConfig
from .dataset import BearingDataset, LengthBatchSampler, TimeGroupBatchSampler
from .evaluation import binary_metrics, select_f1_threshold
from .model import BearingDiagnosisModel, model_metadata, ssl_consistency_loss, topk_mil_loss
from .schemas import SampleRecord, write_jsonl
from .splitting import assert_no_group_leakage
from .training import (
    _model_inputs,
    checkpoint_improved,
    collate_same_length,
    fit_preprocessing,
    resolve_training_device,
    set_seed,
    update_convergence_count,
)


def _augment_waveform(waveform: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    result = waveform.clone()
    max_shift = int(result.shape[-1] * config.ssl_shift_fraction)
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (result.shape[0],), device=result.device)
        result = torch.stack([torch.roll(row, int(shift.item()), dims=-1) for row, shift in zip(result, shifts)])
    scale = result.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return result + torch.randn_like(result) * scale * config.ssl_noise_std_fraction


def _topk_probability(logits: torch.Tensor, fraction: float) -> float:
    k = max(1, math.ceil(logits.numel() * fraction))
    return float(torch.sigmoid(logits.topk(k).values.mean()).detach().cpu())


@torch.no_grad()
def _evaluate_bags(
    model: BearingDiagnosisModel,
    loader: DataLoader,
    device: torch.device,
    config: ModelConfig,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | int]], float]:
    model.eval()
    samples: list[dict[str, float | str | int]] = []
    bags: list[dict[str, float | int]] = []
    losses: list[float] = []
    for bag_index, batch in enumerate(loader):
        output = model(**_model_inputs(batch, device))
        loss, _ = topk_mil_loss(
            output,
            batch["target"].to(device),
            config.mil_top_fraction,
            config.mechanism_aux_weight,
            config.normal_center_weight,
        )
        losses.append(float(loss))
        probability = torch.sigmoid(output["abnormal_logit"]).cpu().numpy()
        auxiliary = torch.sigmoid(output["mechanism_aux_logit"]).cpu().numpy()
        targets = batch["target"].numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            samples.append({
                "sample_id": str(sample_id),
                "target": int(targets[index]),
                "abnormal_probability": float(probability[index]),
                "mechanism_aux_probability": float(auxiliary[index]),
                "normal_distance": float(output["normal_distance"][index].cpu()),
                "q_global": float(output["q_global"][index].cpu()),
                "g_global": float(output["g_global"][index].cpu()),
                "mechanism_to_spectrum_norm_ratio": float(
                    output["mechanism_to_spectrum_norm_ratio"][index].cpu()
                ),
            })
        bags.append({
            "bag_index": bag_index,
            "target": int(targets[0]),
            "sample_count": len(targets),
            "bag_probability": _topk_probability(output["abnormal_logit"], config.mil_top_fraction),
        })
    return samples, bags, float(np.mean(losses)) if losses else float("nan")


def _compute_normal_center(
    model: BearingDiagnosisModel, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.encode_spectrum(
                batch["waveform"].to(device),
                batch["spectrum"].to(device),
                batch["rpm_normalized"].to(device),
            ))
    if not embeddings:
        raise ValueError("normal training samples are required for model v2")
    return torch.cat(embeddings).mean(0)


def train_mil_v2(
    config: ModelConfig,
    train_records: list[SampleRecord],
    validation_records: list[SampleRecord],
    run_dir: str | Path,
    seed: int = 2026,
    device_name: str | None = None,
) -> dict[str, object]:
    if not train_records or not validation_records:
        raise ValueError("non-empty train and validation records are required")
    if config.algorithm_version != "v2_ssl_mil":
        raise ValueError("train_mil_v2 requires algorithm_version=v2_ssl_mil")
    if {r.machine_type for r in train_records + validation_records} != {config.machine_type}:
        raise ValueError("manifest machine_type does not match model configuration")
    set_seed(seed)
    device = resolve_training_device(device_name)
    device_description = (
        f"{device} ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else str(device)
    )
    print(
        f"[setup-v2] model={config.model_name} experiment={config.experiment} seed={seed} "
        f"device={device_description} train={len(train_records)} validation={len(validation_records)}",
        flush=True,
    )
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    assert_no_group_leakage(train_records + validation_records)
    import yaml

    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    write_jsonl(output_dir / "split_manifest.jsonl", [r.to_dict() for r in train_records + validation_records])
    stats = {
        "sample_count": len(train_records) + len(validation_records),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "independent_fault_event_count": len({r.fault_event_id for r in train_records + validation_records if r.fault_event_id}),
    }
    print("[snapshot] freezing training/validation source files...", flush=True)
    write_snapshot(output_dir / "data_snapshot.json", train_records + validation_records, stats)
    write_jsonl(output_dir / "rejected_samples.jsonl", [])
    print("[preprocess] fitting amplitude scale, Hz grid and mechanism scaler...", flush=True)
    preprocess, scaler = fit_preprocessing(train_records, config)
    save_preprocess_state(output_dir, preprocess, scaler)
    print(
        f"[preprocess-done] frequency_bins={preprocess.frequency_grid.axis_hz.size} "
        f"f_max_hz={preprocess.frequency_grid.f_max_hz:g} delta_f_hz={preprocess.frequency_grid.delta_f_hz:g}",
        flush=True,
    )

    train_dataset = BearingDataset(train_records, preprocess, scaler)
    validation_dataset = BearingDataset(validation_records, preprocess, scaler)
    normal_records = [r for r in train_records if not r.is_observed_scope_abnormal]
    normal_dataset = BearingDataset(normal_records, preprocess, scaler)
    normal_loader = DataLoader(
        normal_dataset,
        batch_sampler=LengthBatchSampler(normal_records, config.batch_size),
        collate_fn=collate_same_length,
    )
    train_sampler = TimeGroupBatchSampler(train_records, seed=seed, shuffle=True)
    validation_sampler = TimeGroupBatchSampler(validation_records)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_same_length)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, collate_fn=collate_same_length)
    train_labels = [int(train_records[batch[0]].is_observed_scope_abnormal) for batch in train_sampler.batches]
    bag_counts = {label: train_labels.count(label) for label in (0, 1)}
    if not all(bag_counts.values()):
        raise ValueError("model v2 requires both normal and abnormal training bags")
    bag_weights = {label: len(train_labels) / (2.0 * count) for label, count in bag_counts.items()}
    print(
        f"[sampling-v2] train_bags={len(train_sampler)} normal_bags={bag_counts[0]} "
        f"abnormal_bags={bag_counts[1]} validation_bags={len(validation_sampler)} "
        f"top_fraction={config.mil_top_fraction:g}",
        flush=True,
    )
    (output_dir / "training_sampling.json").write_text(json.dumps({
        "strategy": "complete_time_group_topk_mil",
        "training_sample_count": len(train_records),
        "training_bag_count": len(train_sampler),
        "validation_bag_count": len(validation_sampler),
        "normal_training_bags": bag_counts[0],
        "abnormal_training_bags": bag_counts[1],
        "top_fraction": config.mil_top_fraction,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }, indent=2), encoding="utf-8")

    model = BearingDiagnosisModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ssl_history: list[dict[str, float | int]] = []
    print(f"[stage] normal self-supervised pretraining epochs={config.ssl_pretrain_epochs}", flush=True)
    for epoch in range(1, config.ssl_pretrain_epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for batch_index, batch in enumerate(normal_loader, start=1):
            inputs = _model_inputs(batch, device)
            first = model.encode_spectrum(_augment_waveform(inputs["waveform"], config), inputs["spectrum"], inputs["rpm_normalized"])
            second = model.encode_spectrum(_augment_waveform(inputs["waveform"], config), inputs["spectrum"], inputs["rpm_normalized"])
            loss = ssl_consistency_loss(first, second, config.ssl_variance_weight, config.ssl_covariance_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            if batch_index % 25 == 0 or batch_index == len(normal_loader):
                print(
                    f"[ssl-progress {epoch:03d}/{config.ssl_pretrain_epochs}] "
                    f"batches={batch_index}/{len(normal_loader)}",
                    flush=True,
                )
        value = float(np.mean(losses))
        ssl_history.append({"epoch": epoch, "loss": value})
        print(f"[ssl {epoch:03d}/{config.ssl_pretrain_epochs}] normal_loss={value:.5f} time={time.perf_counter()-started:.1f}s", flush=True)
    model.normal_center.copy_(_compute_normal_center(model, normal_loader, device))
    (output_dir / "ssl_history.json").write_text(json.dumps(ssl_history, indent=2), encoding="utf-8")

    best_state: dict[str, torch.Tensor] | None = None
    best_pr_auc, best_validation_loss, best_epoch = -1.0, float("inf"), 0
    patience = convergence_count = 0
    stop_reason = "max_epochs"
    history: list[dict[str, float | int]] = []
    print(f"[stage] top-k MIL training epochs={config.max_epochs}", flush=True)
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            output = model(**_model_inputs(batch, device))
            label = int(batch["target"][0].item())
            loss, _ = topk_mil_loss(
                output, batch["target"].to(device), config.mil_top_fraction,
                config.mechanism_aux_weight, config.normal_center_weight, bag_weights[label],
            )
            loss.backward()
            accumulated += 1
            losses.append(float(loss.detach()))
            if accumulated == config.gradient_accumulation_steps or batch_index == len(train_loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"[mil-progress {epoch:03d}/{config.max_epochs}] "
                    f"bags={batch_index}/{len(train_loader)}",
                    flush=True,
                )
        validation_predictions, validation_bags, validation_loss = _evaluate_bags(model, validation_loader, device, config)
        targets = [int(row["target"]) for row in validation_bags]
        probabilities = [float(row["bag_probability"]) for row in validation_bags]
        threshold = select_f1_threshold(targets, probabilities)
        metrics = binary_metrics(targets, probabilities, threshold)
        current = float(metrics["pr_auc_range_label"])
        convergence_count = update_convergence_count(validation_loss, config.convergence_val_loss, convergence_count)
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                        "validation_bag_pr_auc": current, "validation_bag_f1": float(metrics["f1"]),
                        "threshold": threshold, "training_bags": len(train_sampler)})
        if checkpoint_improved(current, validation_loss, best_pr_auc, best_validation_loss):
            best_pr_auc, best_validation_loss, best_epoch = current, validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": asdict(config), "evaluation_threshold": threshold,
                        "normal_center": model.normal_center.detach().cpu(),
                        "threshold_unit": "time_group_topk", "best_epoch": best_epoch,
                        "best_validation_pr_auc": best_pr_auc, "best_validation_loss": best_validation_loss,
                        "metadata": model_metadata(model)}, output_dir / "model_best_during_training.pt")
            patience = 0
        else:
            patience += 1
        print(f"[mil {epoch:03d}/{config.max_epochs}] train_loss={train_loss:.5f} val_loss={validation_loss:.5f} bag_pr_auc={current:.4f} bag_f1={float(metrics['f1']):.4f} patience={patience}/{config.early_stopping_patience} time={time.perf_counter()-started:.1f}s", flush=True)
        if convergence_count >= config.convergence_epochs:
            stop_reason = "converged_validation_loss"
            break
        if patience >= config.early_stopping_patience:
            stop_reason = "early_stopping_patience"
            break
    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    validation_predictions, validation_bags, _ = _evaluate_bags(model, validation_loader, device, config)
    targets = [int(row["target"]) for row in validation_bags]
    probabilities = [float(row["bag_probability"]) for row in validation_bags]
    threshold = select_f1_threshold(targets, probabilities)
    metrics = binary_metrics(targets, probabilities, threshold)
    metrics["evaluation_unit"] = "object_id_plus_sample_group_id_topk"
    metrics["bag_count"] = len(validation_bags)
    checkpoint: dict[str, Any] = {"state_dict": best_state, "config": asdict(config),
        "normal_center": model.normal_center.detach().cpu(),
        "evaluation_threshold": threshold, "threshold_unit": "time_group_topk", "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc, "best_validation_loss": best_validation_loss,
        "stop_reason": stop_reason, "metadata": model_metadata(model)}
    torch.save(checkpoint, output_dir / "model.pt")
    save_preprocess_state(output_dir, preprocess, scaler)
    (output_dir / "metrics_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "validation_bags.json").write_text(json.dumps(validation_bags, indent=2), encoding="utf-8")
    write_prediction_parquet(output_dir, validation_predictions, "validation")
    write_model_card(output_dir, config, threshold, metrics)
    print(f"[done-v2] best_epoch={best_epoch} bag_pr_auc={float(metrics['pr_auc_range_label']):.4f} bag_f1={float(metrics['f1']):.4f} threshold={threshold:.6f} run_dir={output_dir}", flush=True)
    return {"run_dir": str(output_dir), "threshold": threshold, "threshold_unit": "time_group_topk", "metrics": metrics}
