from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from bearing_diagnosis.manifest import assign_configured_splits
from bearing_diagnosis.schemas import SampleRecord


def _record(day: int, abnormal: bool) -> SampleRecord:
    timestamp = datetime(2026, 1, 1) + timedelta(days=day)
    return SampleRecord(
        sample_id=f"sample-{abnormal}-{day}",
        object_id="OBJECT",
        project_id="PROJECT",
        turbine_id="TURBINE",
        machine_type="semi_direct",
        sensor_position="3点",
        waveform_path="waveform.csv",
        acquisition_time=timestamp,
        sampling_rate_hz=25600.0,
        waveform_length=1024,
        rpm=200.0,
        rpm_source="filename",
        range_id="OBJECT_p03_abnormal" if abnormal else "OBJECT_p03_normal",
        fault_event_id="event-1" if abnormal else None,
        is_observed_scope_abnormal=abnormal,
        label_source="abnormal_time_range" if abnormal else "confirmed_normal_directory",
        component_orders={
            "rolling_element": 5.0,
            "cage": 0.4,
            "outer_race": 12.0,
            "inner_race": 14.0,
        },
        sample_group_id=f"group-{abnormal}-{day}",
    )


def test_configured_split_policy_can_assign_labels_to_different_roles() -> None:
    records = [_record(day, abnormal) for abnormal in (False, True) for day in range(6)]
    config = {
        "OBJECT": {
            "role": "development",
            "split_policy": {"normal": "train", "abnormal": "external_test"},
        }
    }
    split = assign_configured_splits(records, config)
    assert {item.dataset_split for item in split if not item.is_observed_scope_abnormal} == {"train"}
    assert {item.dataset_split for item in split if item.is_observed_scope_abnormal} == {"external_test"}


def test_configured_split_policy_preserves_development_splitting() -> None:
    records = [_record(day, False) for day in range(6)]
    config = {
        "OBJECT": {
            "role": "development",
            "split_policy": {"normal": "development", "abnormal": "external_test"},
        }
    }
    split = assign_configured_splits(records, config)
    assert {item.dataset_split for item in split} == {"train", "validation"}
