from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

import pytest

from bearing_diagnosis.config import ModelConfig
from bearing_diagnosis.dataset import TimeGroupBatchSampler
from test_framework import record


def test_v2_config_is_separate_from_data_version() -> None:
    config = ModelConfig("data-v2-does-not-imply-model-v2", "dfig")
    assert config.algorithm_version == "v1_sample_bce"
    assert replace(config, algorithm_version="v2_ssl_mil").algorithm_version == "v2_ssl_mil"


def test_time_group_sampler_keeps_complete_bags() -> None:
    records = [
        replace(record(0, False), object_id="a", sample_group_id="g0", dataset_split="train"),
        replace(record(1, False), object_id="a", sample_group_id="g0", dataset_split="train"),
        replace(record(2, True), object_id="b", sample_group_id="g1", dataset_split="train"),
    ]
    sampler = TimeGroupBatchSampler(records, shuffle=False)
    assert list(sampler) == [[0, 1], [2]]


def test_positive_topk_mil_only_backpropagates_through_top_instances() -> None:
    torch = pytest.importorskip("torch")
    from bearing_diagnosis.model import topk_mil_loss

    logits = torch.arange(10.0, requires_grad=True)
    outputs = {
        "abnormal_logit": logits,
        "mechanism_aux_logit": torch.zeros(10),
        "q_global": torch.zeros(10),
        "normal_distance": torch.zeros(10),
    }
    loss, details = topk_mil_loss(
        outputs, torch.ones(10), top_fraction=0.2,
        auxiliary_weight=0.0, normal_center_weight=0.0,
    )
    loss.backward()
    active = torch.nonzero(logits.grad, as_tuple=False).flatten().tolist()
    assert active == [8, 9]
    assert details["normal_center_loss"] == 0.0


def test_normal_mil_constrains_every_instance_and_center() -> None:
    torch = pytest.importorskip("torch")
    from bearing_diagnosis.model import topk_mil_loss

    logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    outputs = {
        "abnormal_logit": logits,
        "mechanism_aux_logit": torch.zeros(3),
        "q_global": torch.zeros(3),
        "normal_distance": torch.ones(3),
    }
    loss, details = topk_mil_loss(outputs, torch.zeros(3), normal_center_weight=0.1)
    loss.backward()
    assert torch.all(logits.grad != 0)
    assert details["normal_center_loss"] == pytest.approx(1.0)


def test_normal_center_does_not_break_historical_state_dict_shape() -> None:
    pytest.importorskip("torch")
    from bearing_diagnosis.model import BearingDiagnosisModel

    model = BearingDiagnosisModel(ModelConfig("v1", "dfig", time_pool_segments=2, spectrum_pool_segments=2))
    assert "normal_center" not in model.state_dict()


def test_v2_one_epoch_training_and_bag_testing(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    from bearing_diagnosis.testing_v2 import test_one_run_v2
    from bearing_diagnosis.training import train_one_run

    fs, length = 2048.0, 512
    times = np.arange(length) / fs
    records = []
    for split, count in (("train", 4), ("validation", 2)):
        for abnormal in (False, True):
            for index in range(count):
                signal = np.sin(2 * np.pi * (70 if abnormal else 17) * times)
                path = tmp_path / f"{split}-{abnormal}-{index}.npy"
                np.save(path, signal)
                records.append(replace(
                    record(index, abnormal), sample_id=path.stem, waveform_path=str(path),
                    sampling_rate_hz=fs, waveform_length=length, machine_type="dfig",
                    object_id=f"object-{abnormal}", dataset_split=split,
                    sample_group_id=f"{split}-bag-{abnormal}-{index // 2}",
                ))
    config = ModelConfig(
        "v2-smoke", "dfig", algorithm_version="v2_ssl_mil",
        batch_size=4, gradient_accumulation_steps=2, ssl_pretrain_epochs=1,
        max_epochs=1, early_stopping_patience=1, time_pool_segments=2,
        spectrum_pool_segments=2, business_f_max_hz=500.0,
    )
    run_dir = tmp_path / "run-v2"
    result = train_one_run(
        config,
        [item for item in records if item.dataset_split == "train"],
        [item for item in records if item.dataset_split == "validation"],
        run_dir, device_name="cpu",
    )
    assert result["threshold_unit"] == "time_group_topk"
    checkpoint = __import__("torch").load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["algorithm_version"] == "v2_ssl_mil"
    assert checkpoint["normal_center"].shape == (64,)
    test_result = test_one_run_v2(
        run_dir,
        [item for item in records if item.dataset_split == "validation"],
        tmp_path / "test-v2", "cpu",
    )
    assert test_result["metrics"]["bag_level"]["sample_count"] == 2
    metrics = json.loads((tmp_path / "test-v2" / "metrics_test.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_threshold_source"] == "frozen_development_validation_time_group_topk"
    assert (tmp_path / "test-v2" / "predictions_test_bags.parquet").is_file()
