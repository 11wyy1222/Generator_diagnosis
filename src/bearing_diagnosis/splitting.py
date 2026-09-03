from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
import random
from typing import Iterable

from .schemas import SampleRecord


def range_position(acquisition_time: datetime, start: datetime, end: datetime) -> str:
    if not start <= acquisition_time <= end or end <= start:
        raise ValueError("acquisition time must lie inside a non-empty range")
    progress = (acquisition_time - start).total_seconds() / (end - start).total_seconds()
    return "early" if progress < 1 / 3 else "middle" if progress < 2 / 3 else "late"


def assign_daily_groups(
    records: Iterable[SampleRecord], range_bounds: dict[str, tuple[datetime, datetime]]
) -> list[SampleRecord]:
    """Create leakage-safe daily keys shared by sensors of the same object/event."""
    grouped: list[SampleRecord] = []
    for record in records:
        position = None
        if record.is_observed_scope_abnormal:
            if record.range_id not in range_bounds:
                raise ValueError(f"missing bounds for abnormal range {record.range_id}")
            position = range_position(record.acquisition_time, *range_bounds[record.range_id])
        shared_scope = record.fault_event_id or f"{record.object_id}_normal"
        date_key = record.acquisition_time.strftime("%Y%m%d")
        group_id = "_".join(part for part in (shared_scope, position, date_key) if part)
        grouped.append(replace(record, sample_group_id=group_id, range_position=position))
    return grouped


def split_development_records(
    records: Iterable[SampleRecord], seed: int = 2026, validation_fraction: float = 0.30
) -> list[SampleRecord]:
    """Split whole time blocks within object/label/range-position strata."""
    records = list(records)
    if any(record.sample_group_id is None for record in records):
        raise ValueError("sample_group_id must be assigned before splitting")
    group_records: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        group_records[str(record.sample_group_id)].append(record)
    group_stratum: dict[str, tuple[str, bool, str | None]] = {}
    for group_id, members in group_records.items():
        strata = {(r.object_id, r.is_observed_scope_abnormal, r.range_position) for r in members}
        if len(strata) != 1:
            raise ValueError(f"group crosses split strata: {group_id}")
        group_stratum[group_id] = next(iter(strata))
    strata_groups: dict[tuple[str, bool, str | None], list[str]] = defaultdict(list)
    for group_id, stratum in group_stratum.items():
        strata_groups[stratum].append(group_id)
    assignment: dict[str, str] = {}
    for stratum in sorted(strata_groups, key=str):
        groups = sorted(strata_groups[stratum])
        if len(groups) < 3:
            raise ValueError(f"at least three non-empty time blocks required for stratum {stratum}")
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(groups)
        n_validation = max(1, round(validation_fraction * len(groups)))
        n_validation = min(n_validation, len(groups) - 2)
        for index, group_id in enumerate(groups):
            assignment[group_id] = "validation" if index < n_validation else "train"
    return [replace(record, dataset_split=assignment[str(record.sample_group_id)]) for record in records]


def assert_no_group_leakage(records: Iterable[SampleRecord]) -> None:
    memberships: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.sample_group_id and record.dataset_split:
            memberships[record.sample_group_id].add(record.dataset_split)
    leaked = [group_id for group_id, splits in memberships.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"sample groups leak across dataset splits: {leaked[:5]}")

