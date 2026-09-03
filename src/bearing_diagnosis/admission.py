from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocessing import load_waveform, validate_waveform
from .schemas import SampleRecord, write_jsonl


def validate_manifest_records(
    records: Sequence[SampleRecord], rejected_path: str | Path | None = None
) -> tuple[list[SampleRecord], list[dict[str, object]], dict[str, object]]:
    """Run non-destructive admission checks and report every rejection."""
    lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    loaded: dict[str, np.ndarray] = {}
    initial_rejections: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if not (Path(record.waveform_path).is_file()):
            initial_rejections[record.sample_id].append("file_missing")
            continue
        try:
            signal = load_waveform(record.waveform_path)
            loaded[record.sample_id] = signal
            lengths[(record.object_id, record.sensor_position)].append(int(signal.size))
            initial_rejections[record.sample_id].extend(validate_waveform(signal))
            if int(signal.size) != record.waveform_length:
                initial_rejections[record.sample_id].append("manifest_waveform_length_conflict")
        except ValueError as exc:
            initial_rejections[record.sample_id].append(f"unreadable:{exc}")
    target_lengths = {
        key: Counter(values).most_common(1)[0][0] for key, values in lengths.items() if values
    }
    accepted: list[SampleRecord] = []
    rejected: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for record in records:
        reasons = list(initial_rejections[record.sample_id])
        if record.sample_id in seen_ids:
            reasons.append("duplicate_sample_id")
        seen_ids.add(record.sample_id)
        expected = target_lengths.get((record.object_id, record.sensor_position))
        signal = loaded.get(record.sample_id)
        if expected is not None and signal is not None and signal.size != expected:
            reasons.append("inconsistent_waveform_length")
        if reasons:
            rejected.append({"sample_id": record.sample_id, "waveform_path": record.waveform_path, "reasons": sorted(set(reasons))})
        else:
            accepted.append(record)
    if rejected_path is not None:
        write_jsonl(rejected_path, rejected)
    stats: dict[str, object] = {
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "independent_fault_event_count": len({r.fault_event_id for r in accepted if r.fault_event_id}),
        "by_object_sensor_range_label": {},
    }
    counts = Counter(
        (r.object_id, r.sensor_position, r.range_id, str(int(r.is_observed_scope_abnormal))) for r in accepted
    )
    stats["by_object_sensor_range_label"] = {
        "|".join(key): value for key, value in sorted(counts.items())
    }
    return accepted, rejected, stats


def write_snapshot(path: str | Path, records: Sequence[SampleRecord], stats: dict[str, object]) -> None:
    import hashlib

    entries = []
    for record in sorted(records, key=lambda item: item.sample_id):
        file_path = Path(record.waveform_path)
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        entries.append({"sample_id": record.sample_id, "path": str(file_path), "size": file_path.stat().st_size, "sha256": digest})
    payload = {"stats": stats, "files": entries}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

