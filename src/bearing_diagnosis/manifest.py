from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from .preprocessing import load_waveform, validate_waveform
from .schemas import SampleRecord, parse_waveform_filename, write_jsonl
from .splitting import assert_no_group_leakage, range_position, split_development_records


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _range_id(object_id: str, sensor_position: str, abnormal: bool) -> str:
    sensor_codes = {
        "12点": "p12", "3点": "p03", "6点": "p06", "9点": "p09",
        "非驱动端": "nde", "驱动端": "de",
    }
    return f"{object_id}_{sensor_codes[sensor_position]}_{'abnormal' if abnormal else 'normal'}"


def _sample_id(object_id: str, sensor_position: str, acquisition_time: datetime) -> str:
    sensor_codes = {
        "12点": "p12", "3点": "p03", "6点": "p06", "9点": "p09",
        "非驱动端": "nde", "驱动端": "de",
    }
    return f"{object_id}_{sensor_codes[sensor_position]}_{acquisition_time:%Y%m%d%H%M%S}"


def _rpm_bin(rpm: float, specification: dict[str, float]) -> str:
    minimum = float(specification["minimum"])
    maximum = float(specification["maximum"])
    width = float(specification["width"])
    if not minimum <= rpm <= maximum:
        raise ValueError(f"RPM {rpm} lies outside bin range [{minimum}, {maximum}]")
    if rpm == maximum:
        lower = maximum - width
        upper = maximum
        right = "]"
    else:
        index = int((rpm - minimum) // width)
        lower = minimum + index * width
        upper = min(lower + width, maximum)
        right = ")" if upper < maximum else "]"
    return f"[{lower:g},{upper:g}{right}"


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _assign_time_groups(
    records: Iterable[SampleRecord], range_bounds: dict[str, tuple[datetime, datetime]]
) -> list[SampleRecord]:
    records = list(records)
    scopes: dict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    for record in records:
        scopes[(record.object_id, record.range_id)].append(record)

    result: list[SampleRecord] = []
    for (_, range_id), members in scopes.items():
        timestamps = sorted({member.acquisition_time for member in members})
        positive_gaps = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        median_gap = statistics.median(positive_gaps) if positive_gaps else None
        batch_by_time: dict[datetime, int] = {}
        batch = 1
        previous: datetime | None = None
        for timestamp in timestamps:
            if (
                previous is not None
                and timestamp.date() == previous.date()
                and median_gap is not None
                and (timestamp - previous).total_seconds() > 3 * median_gap
            ):
                batch += 1
            elif previous is None or timestamp.date() != previous.date():
                batch = 1
            batch_by_time[timestamp] = batch
            previous = timestamp

        daily_batch_counts = Counter((timestamp.date(), batch_by_time[timestamp]) for timestamp in timestamps)
        batches_per_day = Counter(day for day, _ in daily_batch_counts)
        for member in members:
            position = None
            if member.is_observed_scope_abnormal:
                position = range_position(member.acquisition_time, *range_bounds[range_id])
            shared_scope = member.fault_event_id or f"{member.object_id}_normal"
            parts = [shared_scope]
            if position:
                parts.append(position)
            parts.append(member.acquisition_time.strftime("%Y%m%d"))
            batch_number = batch_by_time[member.acquisition_time]
            if batches_per_day[member.acquisition_time.date()] > 1:
                parts.append(f"b{batch_number:02d}")
            result.append(
                replace(member, sample_group_id="_".join(parts), range_position=position)
            )
    return result


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _interval_statistics(records: Iterable[SampleRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[datetime]] = defaultdict(list)
    for record in records:
        groups[(record.object_id, record.sensor_position, record.range_id)].append(record.acquisition_time)
    output: list[dict[str, Any]] = []
    for (object_id, sensor, range_id), values in sorted(groups.items()):
        times = sorted(values)
        gaps = sorted(
            (right - left).total_seconds() for left, right in zip(times, times[1:]) if right > left
        )
        median_gap = statistics.median(gaps) if gaps else None
        output.append({
            "object_id": object_id,
            "sensor_position": sensor,
            "range_id": range_id,
            "sample_count": len(times),
            "first_acquisition_time": times[0].isoformat(sep=" "),
            "last_acquisition_time": times[-1].isoformat(sep=" "),
            "duplicate_timestamp_count": len(times) - len(set(times)),
            "interval_seconds": {
                "minimum": min(gaps) if gaps else None,
                "median": median_gap,
                "p90": _percentile(gaps, 0.90),
                "maximum": max(gaps) if gaps else None,
            },
            "obvious_gap_count": (
                sum(gap > 3 * median_gap for gap in gaps) if median_gap is not None else 0
            ),
        })
    return output


def build_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    objects_config_path: str | Path,
    sources_config_path: str | Path,
) -> dict[str, Any]:
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    object_config = _read_json(objects_config_path)
    source_config = _read_json(sources_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "manifests"
    splits_dir = output_root / "splits"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    snapshot_paths: set[Path] = set()
    source_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    range_rows: list[dict[str, Any]] = []
    range_bounds: dict[str, tuple[datetime, datetime]] = {}

    for object_id, source in source_config["objects"].items():
        configured = object_config[object_id]
        abnormal_bounds = (
            tuple(map(_time, configured["abnormal_range"]))
            if configured.get("abnormal_range") else None
        )
        normal_bounds = (
            tuple(map(_time, configured["normal_range"])) if configured["normal_range"] else None
        )
        if configured["machine_type"] == "semi_direct":
            bin_spec = source_config["rpm_bins"]["semi_direct"]
        elif object_id == "DFIG_TEST_F14":
            bin_spec = source_config["rpm_bins"]["dfig_f14"]
        else:
            bin_spec = source_config["rpm_bins"]["dfig_development"]

        for directory_spec in source["directories"]:
            directory = raw_root / directory_spec["name"]
            if not directory.is_dir():
                raise FileNotFoundError(f"configured source directory does not exist: {directory}")
            sensor_position = directory_spec["sensor_position"]
            source_label = directory_spec.get("label")
            abnormal_range_id = None
            if abnormal_bounds is not None and source_label != "confirmed_normal":
                abnormal_start, abnormal_end = abnormal_bounds
                abnormal_range_id = _range_id(object_id, sensor_position, True)
                range_bounds[abnormal_range_id] = abnormal_bounds
                range_rows.append({
                    "range_id": abnormal_range_id,
                    "object_id": object_id,
                    "sensor_position": sensor_position,
                    "range_type": "full_abnormal" if source_label == "abnormal" else "core_abnormal",
                    "start": abnormal_start.isoformat(sep=" "),
                    "end": abnormal_end.isoformat(sep=" "),
                    "full_abnormal_start": source.get("full_abnormal_range", [None, None])[0],
                    "full_abnormal_end": source.get("full_abnormal_range", [None, None])[1],
                    "fault_event_id": source.get("fault_event_id"),
                    "known_fault_component": source.get("known_fault_component"),
                })
            normal_range_id = None
            if source_label == "confirmed_normal":
                normal_range_id = _range_id(object_id, sensor_position, False)
                range_rows.append({
                    "range_id": normal_range_id,
                    "object_id": object_id,
                    "sensor_position": sensor_position,
                    "range_type": "confirmed_normal_directory",
                    "start": None,
                    "end": None,
                    "source_directory": str(directory.resolve()),
                    "fault_event_id": None,
                    "known_fault_component": None,
                })
            elif normal_bounds is not None:
                normal_range_id = _range_id(object_id, sensor_position, False)
                range_bounds[normal_range_id] = normal_bounds
                range_rows.append({
                    "range_id": normal_range_id,
                    "object_id": object_id,
                    "sensor_position": sensor_position,
                    "range_type": "normal",
                    "start": normal_bounds[0].isoformat(sep=" "),
                    "end": normal_bounds[1].isoformat(sep=" "),
                    "fault_event_id": None,
                    "known_fault_component": None,
                })

            for path in sorted(directory.glob("*.csv")):
                source_counts[object_id] += 1
                try:
                    parsed = parse_waveform_filename(
                        path,
                        require_position=bool(directory_spec.get(
                            "require_position", configured["machine_type"] == "semi_direct"
                        )),
                    )
                except ValueError as exc:
                    rejected.append({
                        "object_id": object_id,
                        "waveform_path": str(path),
                        "reasons": [f"filename_parse_error:{exc}"],
                    })
                    continue
                if parsed.sensor_position and parsed.sensor_position != sensor_position:
                    rejected.append({
                        "object_id": object_id,
                        "waveform_path": str(path),
                        "reasons": ["sensor_position_conflict"],
                    })
                    continue

                abnormal = bool(
                    abnormal_bounds
                    and source_label != "confirmed_normal"
                    and abnormal_bounds[0] <= parsed.acquisition_time <= abnormal_bounds[1]
                )
                normal = source_label == "confirmed_normal" or bool(
                    source_label != "abnormal"
                    and normal_bounds
                    and normal_bounds[0] <= parsed.acquisition_time <= normal_bounds[1]
                )
                if not abnormal and not normal:
                    exclusion_counts[f"{object_id}|outside_supervised_time_ranges"] += 1
                    continue
                snapshot_paths.add(path.resolve())
                range_id = abnormal_range_id if abnormal else normal_range_id
                assert range_id is not None
                rpm_min, rpm_max = map(float, configured["rpm_range"])
                if not rpm_min <= parsed.rpm <= rpm_max:
                    rejected.append({
                        "object_id": object_id,
                        "sensor_position": sensor_position,
                        "waveform_path": str(path.resolve()),
                        "acquisition_time": parsed.acquisition_time.isoformat(sep=" "),
                        "rpm": parsed.rpm,
                        "range_id": range_id,
                        "reasons": ["rpm_out_of_business_range"],
                    })
                    continue
                try:
                    signal = load_waveform(path)
                    reasons = validate_waveform(signal)
                except (OSError, ValueError) as exc:
                    signal = None
                    reasons = [f"unreadable:{exc}"]
                if reasons:
                    rejected.append({
                        "object_id": object_id,
                        "sensor_position": sensor_position,
                        "waveform_path": str(path.resolve()),
                        "acquisition_time": parsed.acquisition_time.isoformat(sep=" "),
                        "rpm": parsed.rpm,
                        "range_id": range_id,
                        "reasons": sorted(set(reasons)),
                    })
                    continue
                assert signal is not None
                pending.append({
                    "sample_id": _sample_id(object_id, sensor_position, parsed.acquisition_time),
                    "object_id": object_id,
                    "project_id": source["project_id"],
                    "turbine_id": source["turbine_id"],
                    "machine_type": configured["machine_type"],
                    "sensor_position": sensor_position,
                    "waveform_path": str(path.resolve()),
                    "acquisition_time": parsed.acquisition_time,
                    "sampling_rate_hz": parsed.sampling_rate_hz,
                    "waveform_length": int(signal.size),
                    "rpm": parsed.rpm,
                    "rpm_source": "filename",
                    "range_id": range_id,
                    "fault_event_id": source["fault_event_id"] if abnormal else None,
                    "is_observed_scope_abnormal": abnormal,
                    "label_source": (
                        "abnormal_time_range" if abnormal
                        else "confirmed_normal_directory" if source_label == "confirmed_normal"
                        else "normal_time_range"
                    ),
                    "component_orders": configured.get("component_orders"),
                    "rpm_bin": _rpm_bin(parsed.rpm, bin_spec),
                })
                if len(pending) % 50 == 0:
                    print(f"validated {len(pending)} candidate waveforms", flush=True)

    lengths: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for raw in pending:
        lengths[(raw["object_id"], raw["sensor_position"])][raw["waveform_length"]] += 1
    target_lengths: dict[tuple[str, str], int] = {}
    for key, counts in lengths.items():
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            raise RuntimeError(f"waveform-length mode is tied for {key}: {dict(counts)}")
        target_lengths[key] = most_common[0][0]

    records: list[SampleRecord] = []
    seen_ids: set[str] = set()
    for raw in pending:
        reasons: list[str] = []
        if raw["waveform_length"] != target_lengths[(raw["object_id"], raw["sensor_position"])]:
            reasons.append("inconsistent_waveform_length")
        if raw["sample_id"] in seen_ids:
            reasons.append("duplicate_sample_id")
        seen_ids.add(raw["sample_id"])
        if reasons:
            rejected.append({
                "sample_id": raw["sample_id"],
                "object_id": raw["object_id"],
                "sensor_position": raw["sensor_position"],
                "waveform_path": raw["waveform_path"],
                "waveform_length": raw["waveform_length"],
                "target_waveform_length": target_lengths[(raw["object_id"], raw["sensor_position"])],
                "reasons": reasons,
            })
        else:
            records.append(SampleRecord(**raw))

    grouped_records = _assign_time_groups(records, range_bounds)
    development_ids = {
        object_id for object_id, config in object_config.items() if config["role"] == "development"
    }
    development = split_development_records(
        [record for record in grouped_records if record.object_id in development_ids], seed=2026
    )
    development_by_id = {record.sample_id: record for record in development}
    split_records: list[SampleRecord] = []
    for record in grouped_records:
        if record.sample_id in development_by_id:
            split_records.append(development_by_id[record.sample_id])
        else:
            role = str(object_config[record.object_id]["role"])
            if role == "development":
                raise RuntimeError(f"development split missing for {record.object_id}")
            split_records.append(replace(record, dataset_split=role))
    assert_no_group_leakage(split_records)

    write_jsonl(manifests_dir / "samples.jsonl", [record.to_dict() for record in grouped_records])
    write_jsonl(manifests_dir / "ranges.jsonl", range_rows)
    write_jsonl(manifests_dir / "rejected_samples.jsonl", rejected)
    write_jsonl(splits_dir / "weak_supervised_split.jsonl", [record.to_dict() for record in split_records])

    group_rows: list[dict[str, Any]] = []
    grouped_split: dict[tuple[str, str, str], list[SampleRecord]] = defaultdict(list)
    for record in split_records:
        grouped_split[(str(record.sample_group_id), record.range_id, str(record.dataset_split))].append(record)
    for (group_id, range_id, split), members in sorted(grouped_split.items()):
        group_rows.append({
            "sample_group_id": group_id,
            "range_id": range_id,
            "fault_event_id": members[0].fault_event_id,
            "range_position": members[0].range_position,
            "dataset_split": split,
            "split_seed": 2026,
            "sample_count": len(members),
            "sample_ids": sorted(member.sample_id for member in members),
        })
    write_jsonl(splits_dir / "split_manifest.jsonl", group_rows)

    accepted_counts = Counter(
        (record.object_id, record.sensor_position, record.range_id, record.dataset_split, record.rpm_bin)
        for record in split_records
    )
    rejection_counts = Counter(
        reason for row in rejected for reason in row.get("reasons", [])
    )
    statistics_payload = {
        "raw_source_file_count": sum(source_counts.values()),
        "source_file_count_by_object": dict(sorted(source_counts.items())),
        "excluded_outside_supervised_time_ranges": dict(sorted(exclusion_counts.items())),
        "accepted_count": len(split_records),
        "rejected_count": len(rejected),
        "rejection_count_by_reason": dict(sorted(rejection_counts.items())),
        "independent_fault_event_count": len({
            record.fault_event_id for record in split_records if record.fault_event_id
        }),
        "target_waveform_length_by_object_sensor": {
            "|".join(key): value for key, value in sorted(target_lengths.items())
        },
        "accepted_count_by_object_sensor_range_split_rpm_bin": {
            "|".join(str(part) for part in key): value for key, value in sorted(accepted_counts.items())
        },
        "interval_statistics": _interval_statistics(split_records),
        "time_block_count": len({record.sample_group_id for record in split_records}),
    }
    (manifests_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    snapshot_entries = []
    for index, path in enumerate(sorted(snapshot_paths), start=1):
        stat = path.stat()
        snapshot_entries.append({
            "path": str(path),
            "size": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
            "sha256": _stream_sha256(path),
        })
        if index % 100 == 0:
            print(f"hashed {index}/{len(snapshot_paths)} in-scope source files", flush=True)
    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(),
        "raw_root": str(raw_root),
        "scope": "all files inside confirmed normal/core-abnormal time ranges before RPM admission",
        "file_count": len(snapshot_entries),
        "total_bytes": sum(entry["size"] for entry in snapshot_entries),
        "objects_config_sha256": _stream_sha256(Path(objects_config_path).resolve()),
        "sources_config_sha256": _stream_sha256(Path(sources_config_path).resolve()),
        "files": snapshot_entries,
    }
    (manifests_dir / "data_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "accepted_count": len(split_records),
        "rejected_count": len(rejected),
        "snapshot_file_count": len(snapshot_entries),
        "time_block_count": statistics_payload["time_block_count"],
    }
