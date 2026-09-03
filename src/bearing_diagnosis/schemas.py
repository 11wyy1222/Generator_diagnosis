from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

COMPONENTS = ("outer_race", "inner_race", "rolling_element", "cage")
SEMI_DIRECT_POSITIONS = ("12点", "3点", "6点", "9点")
_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<fs>[0-9]+(?:\.[0-9]+)?)(?:Hz)?_"
    r"(?P<rpm>[0-9]+(?:\.[0-9]+)?)(?:RPM)?_(?P<time>[0-9]{14})$",
    re.IGNORECASE,
)
_POSITION_RE = re.compile(r"发电机(12点|3点|6点|9点)")


@dataclass(frozen=True)
class ParsedFilename:
    sampling_rate_hz: float
    rpm: float
    acquisition_time: datetime
    sensor_position: str | None


def parse_waveform_filename(path: str | Path, require_position: bool = False) -> ParsedFilename:
    stem = Path(path).stem
    match = _NAME_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"filename does not match required suffix: {Path(path).name}")
    positions = _POSITION_RE.findall(match.group("prefix"))
    if len(positions) > 1:
        raise ValueError(f"multiple generator sensor positions found: {Path(path).name}")
    if require_position and len(positions) != 1:
        raise ValueError(f"exactly one generator sensor position is required: {Path(path).name}")
    return ParsedFilename(
        sampling_rate_hz=float(match.group("fs")),
        rpm=float(match.group("rpm")),
        acquisition_time=datetime.strptime(match.group("time"), "%Y%m%d%H%M%S"),
        sensor_position=positions[0] if positions else None,
    )


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    object_id: str
    project_id: str
    turbine_id: str
    machine_type: str
    sensor_position: str
    waveform_path: str
    acquisition_time: datetime
    sampling_rate_hz: float
    waveform_length: int
    rpm: float
    rpm_source: str
    range_id: str
    fault_event_id: str | None
    is_observed_scope_abnormal: bool
    label_source: str
    component_orders: dict[str, float] | None
    sample_group_id: str | None = None
    range_position: str | None = None
    dataset_split: str | None = None
    rpm_bin: str | None = None

    def __post_init__(self) -> None:
        if self.machine_type not in {"semi_direct", "dfig"}:
            raise ValueError(f"invalid machine_type: {self.machine_type}")
        if self.label_source not in {
            "normal_time_range", "abnormal_time_range", "confirmed_normal_directory"
        }:
            raise ValueError(f"invalid label_source: {self.label_source}")
        if self.sampling_rate_hz <= 0 or self.waveform_length < 4 or self.rpm <= 0:
            raise ValueError("sampling rate, waveform length and RPM must be positive")
        if self.component_orders is not None:
            if set(self.component_orders) != set(COMPONENTS):
                raise ValueError(f"component_orders must contain exactly {COMPONENTS}")
            if not all(math.isfinite(float(v)) and float(v) > 0 for v in self.component_orders.values()):
                raise ValueError("all component orders must be finite positive values")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SampleRecord":
        fields = dict(raw)
        fields["acquisition_time"] = datetime.fromisoformat(str(fields["acquisition_time"]))
        fields.pop("component_labels", None)
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["acquisition_time"] = self.acquisition_time.isoformat(sep=" ")
        result["component_labels"] = {name: None for name in COMPONENTS}
        return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
