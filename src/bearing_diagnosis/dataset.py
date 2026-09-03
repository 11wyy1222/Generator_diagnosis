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
    ) -> None:
        self.records = list(records)
        self.preprocess = preprocess
        self.mechanism_scaler = mechanism_scaler or MechanismScaler(
            np.zeros(FEATURE_COUNT, dtype=np.float32), np.ones(FEATURE_COUNT, dtype=np.float32)
        )
        self.allow_missing_mechanism = allow_missing_mechanism
        # Training revisits the same immutable source waveform every epoch.
        # Cache the fully prepared CPU tensors after their first use so later
        # epochs do not repeatedly parse large CSVs and recompute both FFTs.
        # DataLoader uses no worker processes in the current pipeline, making
        # this a run-local, deterministic cache with no synchronization needs.
        self._prepared_cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        cached = self._prepared_cache.get(index)
        if cached is not None:
            return cached
        record = self.records[index]
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
        }
        self._prepared_cache[index] = prepared
        return prepared


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


class BalancedGroupBatchSampler(Sampler[list[int]]):
    """Balance labels, sensors, abnormal stages and time blocks by length."""

    def __init__(self, records: Sequence[SampleRecord], batch_size: int, seed: int = 2026) -> None:
        if batch_size < 2:
            raise ValueError("balanced batches require batch_size >= 2")
        self.records = list(records)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._batches = self._build(seed)
        if not self._batches:
            raise ValueError("cannot form label-balanced same-length batches")

    def _build(self, seed: int) -> list[list[int]]:
        from collections import defaultdict
        import random

        cells: dict[tuple[int, int, str, str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            stage = str(record.range_position) if record.is_observed_scope_abnormal else "normal"
            key = (
                record.waveform_length,
                int(record.is_observed_scope_abnormal),
                stage,
                record.sensor_position,
                str(record.sample_group_id),
            )
            cells[key].append(index)
        rng = random.Random(seed)
        by_length_stage_sensor: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        # Equal quota per time-block/sensor cell prevents dense acquisition periods dominating.
        by_length: dict[int, list[list[int]]] = defaultdict(list)
        for (length, _, _, _, _), indices in cells.items():
            by_length[length].append(indices)
        quotas = {length: min(len(indices) for indices in groups) for length, groups in by_length.items()}
        for (length, _, stage, sensor, _), indices in cells.items():
            chosen = list(indices)
            rng.shuffle(chosen)
            by_length_stage_sensor[(length, stage, sensor)].extend(chosen[: quotas[length]])
        batches: list[list[int]] = []
        half = self.batch_size // 2
        for length in sorted({key[0] for key in by_length_stage_sensor}):
            sensors = sorted({key[2] for key in by_length_stage_sensor if key[0] == length})
            required = [
                (length, stage, sensor)
                for sensor in sensors
                for stage in ("normal", "early", "middle", "late")
            ]
            if all(by_length_stage_sensor.get(key) for key in required):
                # Each sensor contributes q samples to every abnormal stage and
                # 3q normal samples.  This makes early/middle/late equal while
                # preserving overall normal/abnormal 1:1 and sensor balance.
                quota = min(
                    min(len(by_length_stage_sensor[(length, stage, sensor)]) for sensor in sensors for stage in ("early", "middle", "late")),
                    min(len(by_length_stage_sensor[(length, "normal", sensor)]) // 3 for sensor in sensors),
                )
                negative = []
                positive = []
                for sensor in sensors:
                    normal = list(by_length_stage_sensor[(length, "normal", sensor)])
                    rng.shuffle(normal)
                    negative.extend(normal[: 3 * quota])
                    for stage in ("early", "middle", "late"):
                        staged = list(by_length_stage_sensor[(length, stage, sensor)])
                        rng.shuffle(staged)
                        positive.extend(staged[:quota])
            else:
                # Backward-compatible fallback for fixtures or datasets that do
                # not contain all three abnormal stages.
                negative = []
                positive = []
                for (item_length, stage, _), indices in by_length_stage_sensor.items():
                    if item_length != length:
                        continue
                    (negative if stage == "normal" else positive).extend(indices)
            # Pair the two labels in RPM order to keep their operating-condition
            # distributions as close as the available weak-label data permits.
            negative.sort(key=lambda index: (self.records[index].rpm, rng.random()))
            positive.sort(key=lambda index: (self.records[index].rpm, rng.random()))
            usable = min(len(negative), len(positive))
            for start in range(0, usable, half):
                left, right = negative[start : start + half], positive[start : start + half]
                take = min(len(left), len(right))
                if take:
                    batch = left[:take] + right[:take]
                    rng.shuffle(batch)
                    batches.append(batch)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


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
