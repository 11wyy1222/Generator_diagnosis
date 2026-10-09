from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


AUXILIARY_TRAIN_OBJECTS = {
    "DFIG_NEW_XCH_32",
    "DFIG_NEW_NH_36",
    "DFIG_TEST_F14",
}
MIXED_OBJECTS = {
    "rushan41": "DFIG_DEV_41",
    "nh12": "DFIG_NEW_NH_12",
}
EXTERNAL_SPLITS = {
    "DFIG_NEW_BDB_A1": "external_normal_test",
    "DFIG_NEW_WW_F1": "external_test",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def remap_fold(
    source_rows: list[dict[str, Any]], *, validation_object: str
) -> list[dict[str, Any]]:
    mixed_train_object = next(
        object_id for object_id in MIXED_OBJECTS.values() if object_id != validation_object
    )
    train_objects = AUXILIARY_TRAIN_OBJECTS | {mixed_train_object}
    known_objects = train_objects | {validation_object} | set(EXTERNAL_SPLITS)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        object_id = str(row["object_id"])
        if object_id not in known_objects:
            raise ValueError(f"unexpected object_id: {object_id}")
        if object_id in train_objects:
            row["dataset_split"] = "train"
        elif object_id == validation_object:
            row["dataset_split"] = "validation"
        else:
            expected = EXTERNAL_SPLITS[object_id]
            if source.get("dataset_split") != expected:
                raise ValueError(
                    f"external split changed for {object_id}: "
                    f"{source.get('dataset_split')} != {expected}"
                )
            row["dataset_split"] = expected
        rows.append(row)
    return rows


def summarize_and_validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    waveform_paths = [str(row["waveform_path"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id in output")
    if len(waveform_paths) != len(set(waveform_paths)):
        raise ValueError("duplicate waveform_path in output")

    train = [row for row in rows if row["dataset_split"] == "train"]
    validation = [row for row in rows if row["dataset_split"] == "validation"]
    train_objects = {str(row["object_id"]) for row in train}
    validation_objects = {str(row["object_id"]) for row in validation}
    object_overlap = sorted(train_objects & validation_objects)
    if object_overlap:
        raise ValueError(f"train/validation object overlap: {object_overlap}")

    train_events = {str(row["fault_event_id"]) for row in train if row.get("fault_event_id")}
    validation_events = {
        str(row["fault_event_id"]) for row in validation if row.get("fault_event_id")
    }
    event_overlap = sorted(train_events & validation_events)
    if event_overlap:
        raise ValueError(f"train/validation fault-event overlap: {event_overlap}")

    validation_labels = {bool(row["is_observed_scope_abnormal"]) for row in validation}
    if validation_labels != {False, True}:
        raise ValueError("validation must contain both normal and abnormal samples")

    counts = Counter(
        (str(row["dataset_split"]), bool(row["is_observed_scope_abnormal"]))
        for row in rows
    )
    by_object_split_label = Counter(
        (
            str(row["object_id"]),
            str(row["dataset_split"]),
            bool(row["is_observed_scope_abnormal"]),
        )
        for row in rows
    )
    return {
        "sample_count": len(rows),
        "split_label_counts": [
            {"split": split, "abnormal": abnormal, "count": count}
            for (split, abnormal), count in sorted(counts.items())
        ],
        "by_object_split_label": [
            {
                "object_id": object_id,
                "split": split,
                "abnormal": abnormal,
                "count": count,
            }
            for (object_id, split, abnormal), count in sorted(by_object_split_label.items())
        ],
        "train_objects": sorted(train_objects),
        "validation_objects": sorted(validation_objects),
        "train_validation_object_overlap": object_overlap,
        "train_validation_fault_event_overlap": event_overlap,
        "duplicate_sample_id_count": len(sample_ids) - len(set(sample_ids)),
        "duplicate_waveform_path_count": len(waveform_paths) - len(set(waveform_paths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build two DFIG whole-turbine development folds."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    if not source_rows:
        raise ValueError("source manifest is empty")
    source_ids = [str(row["sample_id"]) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate sample_id in source manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "fold_a_validate_nh12": MIXED_OBJECTS["nh12"],
        "fold_b_validate_rushan41": MIXED_OBJECTS["rushan41"],
    }
    outputs: dict[str, Any] = {}
    for name, validation_object in definitions.items():
        rows = remap_fold(source_rows, validation_object=validation_object)
        if [row["sample_id"] for row in rows] != source_ids:
            raise AssertionError("output row identity/order differs from source")
        summary = summarize_and_validate(rows)
        output_path = args.output_dir / f"dfig_twofold_{name}_server.jsonl"
        digest = write_jsonl(output_path, rows)
        outputs[name] = {
            "path": str(output_path.resolve()),
            "sha256": digest,
            "validation_object": validation_object,
            **summary,
        }

    metadata = {
        "protocol": "dfig_twofold_whole_turbine_development_v1",
        "source_manifest": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "rules": {
            "whole_object_train_validation_isolation": True,
            "fault_event_train_validation_isolation": True,
            "mixed_label_validation_required": True,
            "bdb_a1_role": "external_normal_test",
            "wuwei_f1_role": "external_test",
            "final_acceptance": "future untouched mixed-label turbine or time period",
        },
        "folds": outputs,
    }
    metadata_path = args.output_dir / "dfig_twofold_manifest_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
