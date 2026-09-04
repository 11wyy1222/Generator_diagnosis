from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required; run: pip install -r requirements.txt") from exc

from .mechanism import FEATURE_COUNT, extract_mechanism_features
from .preprocessing import PreprocessState, load_waveform, native_spectra, validate_waveform
from .schemas import SampleRecord


@dataclass(frozen=True)
class MechanismScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, masks: np.ndarray) -> "MechanismScaler":
        values = np.asarray(features, dtype=np.float64)
        valid = np.asarray(masks, dtype=bool)
        if values.ndim != 3 or values.shape != valid.shape:
            raise ValueError("mechanism features and masks must have shape [samples, 4, features]")
        center = np.zeros(values.shape[-1], dtype=np.float64)
        scale = np.ones(values.shape[-1], dtype=np.float64)
        for index in range(values.shape[-1]):
            selected = values[..., index][valid[..., index]]
            if selected.size:
                center[index] = np.median(selected)
                q25, q75 = np.percentile(selected, [25, 75])
                scale[index] = max(float(q75 - q25), 1e-8)
        return cls(center=center.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(features) - self.center) / self.scale
        return np.where(mask, np.clip(scaled, -20.0, 20.0), 0.0).astype(np.float32)


class BearingDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        preprocess: PreprocessState,
        mechanism_scaler: MechanismScaler | None = None,
        allow_missing_mechanism: bool = False,
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        self.records = list(records)
        self.preprocess = preprocess
        self.mechanism_scaler = mechanism_scaler or MechanismScaler(
            np.zeros(FEATURE_COUNT, dtype=np.float32), np.ones(FEATURE_COUNT, dtype=np.float32)
        )
        self.allow_missing_mechanism = allow_missing_mechanism
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.records), dtype=np.float32)
        else:
            self.sample_weights = np.asarray(sample_weights, dtype=np.float32)
            if self.sample_weights.shape != (len(self.records),):
                raise ValueError("sample_weights must contain one value per record")
            if not np.isfinite(self.sample_weights).all() or np.any(self.sample_weights < 0):
                raise ValueError("sample_weights must be finite and non-negative")
        # Training revisits the same immutable source waveform every epoch.
        # Cache the fully prepared CPU tensors after their first use so later
        # epochs do not repeatedly parse large CSVs and recompute both FFTs.
        # DataLoader uses no worker processes in the current pipeline, making
        # this a run-local, deterministic cache with no synchronization needs.
        self._prepared_cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, object]:
        if isinstance(index, tuple):
            record_index, coverage_weight = int(index[0]), float(index[1])
        else:
            record_index, coverage_weight = int(index), 1.0
        cached = self._prepared_cache.get(record_index)
        if cached is not None:
            if coverage_weight == 1.0:
                return cached
            weighted = dict(cached)
            weighted["loss_weight"] = cached["loss_weight"] * coverage_weight
            return weighted
        record = self.records[record_index]
        signal = load_waveform(record.waveform_path)
        reasons = validate_waveform(signal, record.waveform_length)
        if reasons:
            raise ValueError(f"rejected waveform {record.sample_id}: {', '.join(reasons)}")
        frequency, ordinary, envelope = native_spectra(signal, record.sampling_rate_hz)
        spectrum = self.preprocess.frequency_grid.transform(frequency, ordinary, envelope)
        if record.component_orders is None:
            if not self.allow_missing_mechanism:
                raise ValueError(
                    f"component orders are unconfirmed for {record.object_id}; "
                    "only a spectrum-only model may evaluate this record"
                )
            raw_features = np.zeros((4, FEATURE_COUNT), dtype=np.float32)
            valid_mask = np.zeros((4, FEATURE_COUNT), dtype=bool)
            q_global = 0.0
        else:
            evidence = extract_mechanism_features(
                frequency,
                ordinary,
                envelope,
                record.rpm,
                record.component_orders,
                record.sampling_rate_hz,
                record.waveform_length,
            )
            raw_features = evidence.features
            valid_mask = evidence.valid_mask
            q_global = evidence.q_global
        features = self.mechanism_scaler.transform(raw_features, valid_mask)
        prepared: dict[str, object] = {
            "sample_id": record.sample_id,
            "waveform": torch.from_numpy(self.preprocess.normalize_time(signal)[None, :]),
            "spectrum": torch.from_numpy(spectrum),
            "rpm_normalized": torch.tensor(self.preprocess.normalize_rpm(record.rpm), dtype=torch.float32),
            "mechanism_features": torch.from_numpy(features),
            "mechanism_valid_mask": torch.from_numpy(valid_mask),
            "q_global": torch.tensor(q_global, dtype=torch.float32),
            "target": torch.tensor(float(record.is_observed_scope_abnormal), dtype=torch.float32),
            "loss_weight": torch.tensor(self.sample_weights[record_index], dtype=torch.float32),
        }
        self._prepared_cache[record_index] = prepared
        if coverage_weight == 1.0:
            return prepared
        weighted = dict(prepared)
        weighted["loss_weight"] = prepared["loss_weight"] * coverage_weight
        return weighted


def collect_raw_mechanism_features(
    records: Sequence[SampleRecord], preprocess: PreprocessState
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records:
        if record.component_orders is None:
            raise ValueError(f"component orders are unconfirmed for {record.object_id}")
        signal = load_waveform(record.waveform_path)
        frequency, ordinary, envelope = native_spectra(signal, record.sampling_rate_hz)
        evidence = extract_mechanism_features(
            frequency, ordinary, envelope, record.rpm, record.component_orders,
            record.sampling_rate_hz, record.waveform_length,
        )
        features.append(evidence.features)
        masks.append(evidence.valid_mask)
    return np.stack(features), np.stack(masks)


def records_from_manifest(
    path: str | Path,
    split: str | None = None,
    machine_type: str | None = None,
) -> list[SampleRecord]:
    from .schemas import read_jsonl

    records = [SampleRecord.from_dict(raw) for raw in read_jsonl(path)]
    return [
        record
        for record in records
        if (split is None or record.dataset_split == split)
        and (machine_type is None or record.machine_type == machine_type)
    ]


def group_class_loss_weights(records: Sequence[SampleRecord]) -> np.ndarray:
    """Give every class equal mass and every time group equal mass within its class."""
    from collections import defaultdict

    grouped: dict[int, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        label = int(record.is_observed_scope_abnormal)
        group_id = str(record.sample_group_id or record.sample_id)
        grouped[label][(record.object_id, group_id)].append(index)
    if not grouped:
        raise ValueError("training records are required to compute loss weights")
    weights = np.zeros(len(records), dtype=np.float64)
    class_mass = 1.0 / len(grouped)
    for groups in grouped.values():
        group_mass = class_mass / len(groups)
        for indices in groups.values():
            weights[indices] = group_mass / len(indices)
    weights *= len(records) / weights.sum()
    return weights.astype(np.float32)


class FullCoverageBatchSampler(Sampler[list[int | tuple[int, float]]]):
    """Use every record once per epoch and zero-weight only the fixed-batch padding."""

    def __init__(self, records: Sequence[SampleRecord], batch_size: int, seed: int = 2026) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not records:
            raise ValueError("training records are required")
        self.records = list(records)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed: int) -> list[list[int | tuple[int, float]]]:
        from collections import defaultdict
        import random

        buckets: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            buckets[record.waveform_length].append(index)
        rng = random.Random(seed)
        batches: list[list[int | tuple[int, float]]] = []
        for length in sorted(buckets):
            indices = list(buckets[length])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch: list[int | tuple[int, float]] = list(
                    indices[start : start + self.batch_size]
                )
                if len(batch) < self.batch_size:
                    padding = list(indices)
                    rng.shuffle(padding)
                    offset = 0
                    while len(batch) < self.batch_size:
                        batch.append((padding[offset % len(padding)], 0.0))
                        offset += 1
                batches.append(batch)
        rng.shuffle(batches)
        return batches

    @property
    def padding_count(self) -> int:
        return sum(isinstance(index, tuple) for batch in self._batches for index in batch)

    @property
    def optimization_steps(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


# Import compatibility for callers of the previous sampler name.  Its behavior is
# intentionally changed to full coverage; it no longer performs balanced downsampling.
BalancedGroupBatchSampler = FullCoverageBatchSampler


class LengthBatchSampler(Sampler[list[int]]):
    """Deterministic evaluation batches that never mix waveform lengths."""

    def __init__(self, records: Sequence[SampleRecord], batch_size: int) -> None:
        from collections import defaultdict

        buckets: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            buckets[record.waveform_length].append(index)
        self.batches = [
            indices[start : start + batch_size]
            for length, indices in sorted(buckets.items())
            for start in range(0, len(indices), batch_size)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
