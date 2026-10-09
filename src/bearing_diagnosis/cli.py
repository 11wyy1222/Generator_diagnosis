from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generator bearing weak-label diagnosis")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest", help="validate waveform admission without changing source data")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output-dir", required=True)
    build = sub.add_parser("build-manifest", help="build frozen weak-label manifests from the confirmed raw data")
    build.add_argument("--raw-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--objects-config", default="configs/objects.json")
    build.add_argument("--sources-config", default="configs/data_sources.json")
    split = sub.add_parser(
        "build-split", help="build a model-specific split from an admitted sample manifest"
    )
    split.add_argument("--manifest", required=True)
    split.add_argument("--objects-config", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--group-output", required=True)
    split.add_argument("--seed", type=int, default=2026)
    test = sub.add_parser("test", help="evaluate one frozen model on a manifest test split")
    test.add_argument("--run-dir", required=True)
    test.add_argument("--manifest", required=True)
    test.add_argument("--split", required=True)
    test.add_argument("--output-dir", required=True)
    test.add_argument("--device")
    test.add_argument(
        "--object-id",
        action="append",
        help="optionally restrict evaluation to one or more object IDs",
    )
    calibrate = sub.add_parser(
        "calibrate-threshold",
        help="create a new run copy with a validation-midpoint evaluation threshold",
    )
    calibrate.add_argument("--run-dir", required=True)
    calibrate.add_argument("--output-dir", required=True)
    freeze = sub.add_parser(
        "freeze-run", help="hash and freeze one v1 model, preprocessing state, and feature pipeline"
    )
    freeze.add_argument("--run-dir", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--source-root", default=".")
    train = sub.add_parser("train", help="train one frozen manifest split")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument(
        "--device",
        default="cuda",
        help="training device (default: cuda; fails instead of falling back to CPU)",
    )
    anomaly_train = sub.add_parser(
        "train-anomaly", help="fit a normal-only anomaly model on frozen development data"
    )
    anomaly_train.add_argument("--manifest", required=True)
    anomaly_train.add_argument("--run-dir", required=True)
    anomaly_train.add_argument("--machine-type", default="dfig")
    anomaly_test = sub.add_parser(
        "test-anomaly", help="evaluate a frozen anomaly model on one manifest split"
    )
    anomaly_test.add_argument("--run-dir", required=True)
    anomaly_test.add_argument("--manifest", required=True)
    anomaly_test.add_argument("--split", required=True)
    anomaly_test.add_argument("--output-dir", required=True)
    anomaly_test.add_argument("--machine-type", default="dfig")
    target_calibration = sub.add_parser(
        "evaluate-target-calibrated",
        help="calibrate a v1 threshold on early target-normal history and test only later data",
    )
    target_calibration.add_argument("--run-dir", required=True)
    target_calibration.add_argument("--manifest", required=True)
    target_calibration.add_argument("--predictions", required=True)
    target_calibration.add_argument("--output-dir", required=True)
    target_calibration.add_argument("--object-id", required=True)
    target_calibration.add_argument("--freeze-manifest", required=True)
    target_calibration.add_argument("--normal-fraction", type=float, default=.60)
    target_calibration.add_argument("--normal-quantile", type=float, default=.95)
    target_calibration.add_argument("--embargo-hours", type=float, default=24.0)
    infer = sub.add_parser("infer", help="run one waveform inference")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--sample-id", required=True)
    infer.add_argument("--waveform", required=True)
    infer.add_argument("--sampling-rate-hz", required=True, type=float)
    infer.add_argument("--rpm", required=True, type=float)
    infer.add_argument("--orders-json", required=True, help="JSON object or path containing four component orders")
    infer.add_argument("--internal-log")
    return parser


def _orders(value: str) -> dict[str, float]:
    path = Path(value)
    raw = json.loads(path.read_text(encoding="utf-8") if path.is_file() else value)
    return {str(key): float(item) for key, item in raw.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        from .manifest import build_dataset

        result = build_dataset(
            args.raw_root,
            args.output_root,
            args.objects_config,
            args.sources_config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-split":
        from .manifest import build_split_manifest

        result = build_split_manifest(
            args.manifest,
            args.objects_config,
            args.output,
            args.group_output,
            args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "test":
        from .dataset import records_from_manifest
        from .testing import load_run_config, test_one_run

        config = load_run_config(args.run_dir)
        records = records_from_manifest(args.manifest, args.split, config.machine_type)
        if args.object_id:
            selected = set(args.object_id)
            records = [record for record in records if record.object_id in selected]
        if config.algorithm_version == "v2_ssl_mil":
            from .testing_v2 import test_one_run_v2

            result = test_one_run_v2(args.run_dir, records, args.output_dir, args.device)
        else:
            result = test_one_run(args.run_dir, records, args.output_dir, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "calibrate-threshold":
        from .calibration import calibrate_threshold

        result = calibrate_threshold(args.run_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "freeze-run":
        from .freeze import create_freeze_manifest

        result = create_freeze_manifest(args.run_dir, args.output, args.source_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-manifest":
        from .admission import validate_manifest_records
        from .schemas import SampleRecord, read_jsonl

        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        records = [SampleRecord.from_dict(raw) for raw in read_jsonl(args.manifest)]
        _, _, stats = validate_manifest_records(records, output / "rejected_samples.jsonl")
        (output / "admission_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False))
        return 0 if stats["rejected_count"] == 0 else 2
    if args.command == "train":
        from .config import load_config
        from .dataset import records_from_manifest
        from .training import train_one_run

        config = load_config(args.config)
        train_records = records_from_manifest(args.manifest, "train", config.machine_type)
        validation_records = records_from_manifest(args.manifest, "validation", config.machine_type)
        result = train_one_run(config, train_records, validation_records, args.run_dir, args.seed, args.device)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "train-anomaly":
        from .anomaly_detection import train_anomaly
        from .dataset import records_from_manifest

        records = records_from_manifest(args.manifest, machine_type=args.machine_type)
        development = [record for record in records if record.dataset_split in {"train", "validation"}]
        result = train_anomaly(development, args.run_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "test-anomaly":
        from .anomaly_detection import test_anomaly
        from .dataset import records_from_manifest

        records = records_from_manifest(args.manifest, args.split, args.machine_type)
        result = test_anomaly(args.run_dir, records, args.output_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "evaluate-target-calibrated":
        from .target_calibration import evaluate_target_calibration

        result = evaluate_target_calibration(
            args.run_dir,
            args.manifest,
            args.predictions,
            args.output_dir,
            args.object_id,
            normal_fraction=args.normal_fraction,
            normal_quantile=args.normal_quantile,
            embargo_hours=args.embargo_hours,
            freeze_manifest_path=args.freeze_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from .inference import BearingInference, InferenceInput

    engine = BearingInference(args.run_dir)
    external, internal = engine.predict(
        InferenceInput(args.sample_id, args.waveform, args.sampling_rate_hz, args.rpm, _orders(args.orders_json))
    )
    print(json.dumps(external, ensure_ascii=False))
    if args.internal_log:
        Path(args.internal_log).write_text(json.dumps(internal, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
