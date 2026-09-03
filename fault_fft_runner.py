from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fft.fault_fft import faultjudge_fft
from core.fft.config import DEFAULT_FAULT_FFT_CONFIG
from models.data_models import BearingParams
from utils.database import MySQLConnector


@dataclass
class QueryData:
    time_data: list[str]
    rpm_data: list[float]
    src_path: list[str]


@dataclass
class FaultFFTRunResult:
    project_number: str
    dataname: str
    sql_query: str
    query_data: QueryData
    faltype: dict[str, int]
    generbearing: BearingParams
    maxpoint_en_history: list[dict]
    maxpoint_history: list[dict]
    mid: np.ndarray
    mid_en: np.ndarray
    timedatanow: list[str]
    envelope_flags: np.ndarray
    fft_flags: np.ndarray
    envelope_values: np.ndarray
    fft_values: np.ndarray
    putout_envelop_fau: np.ndarray
    putout_fft_fau: np.ndarray
    putout_envelop_value: np.ndarray
    putout_fft_value: np.ndarray
    plot_times: list[str]


@dataclass
class FaultFFTBatchResult:
    """
    对齐 MATLAB faultjudge_fft_touse 的核心输出：
    MaxPoint_en_history, MaxPoint_history, mid, mid_en, timedatanow。
    其余矩阵是从这些原始输出派生出的 Python 汇总/绘图数据。
    """

    maxpoint_en_history: list[dict]
    maxpoint_history: list[dict]
    mid: np.ndarray
    mid_en: np.ndarray
    timedatanow: list[str]
    envelope_flags: np.ndarray
    fft_flags: np.ndarray
    envelope_values: np.ndarray
    fft_values: np.ndarray


def log_progress(message: str) -> None:
    print(f"[fault_fft_runner] {message}", flush=True)


def fast_read_numeric(file_path: str | Path) -> np.ndarray:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        # 波形文本会在任意位置换行，每行字段数不固定，不能按规则表格解析。
        # 仅提取独立数值，避免把文件名、单位等文本中的数字误当成采样值。
        number_pattern = re.compile(
            rb"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.])"
        )
        matches = number_pattern.findall(path.read_bytes())
        values = np.fromiter((float(value) for value in matches), dtype=float)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=None)
        numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        values = numeric_df.to_numpy(dtype=float).reshape(-1, order="F")
        values = values[np.isfinite(values)]
    else:
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    if values.size == 0:
        raise ValueError(f"文件中不存在有效数值数据: {path}")

    return values



def extract_keyword_value(text: str, keyword: str) -> str:
    start = text.find(keyword)
    if start < 0:
        return ""

    start += len(keyword)
    end = text.find(" ", start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def infer_turbine_tag(dataname: str) -> str:
    turbine_tag = str(dataname).split("_", 1)[0].strip()
    if not turbine_tag:
        raise ValueError(f"无法从 dataname 提取机位号: {dataname}")
    return turbine_tag


def normalize_turbine_tag(value: Any) -> str:
    """
    将机位号标准化，兼容 23# / 023# / 0023# 这类写法。
    无法识别时回退为去空格后的原字符串。
    """
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return str(int(digits))
    return text


def default_config_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return PROJECT_ROOT / "故障数据"


def readsome(
    project_number: str,
    dataname: str,
    base_dir: str | Path | None = None,
) -> tuple[dict[str, int], BearingParams]:
    """
    对应 MATLAB readsome：
    - 从 the_turbine_info21.xlsx 找项目+机位对应的机型
    - 从 test_jixing.xlsx 找机型配置
    - 从 test_bearing.xlsx 找轴承特征频率参数
    """
    config_dir = default_config_dir(base_dir)
    bearing_path = config_dir / "test_bearing.xlsx"
    jixing_path = config_dir / "test_jixing.xlsx"
    turbine_info_path = config_dir / "the_turbine_info21.xlsx"
    log_progress(f"读取配置目录: {config_dir}")

    for path in (bearing_path, jixing_path, turbine_info_path):
        if not path.exists():
            raise FileNotFoundError(f"未找到配置文件: {path}")

    bearing_df = pd.read_excel(bearing_path)
    jixing_df = pd.read_excel(jixing_path)
    turbine_df = pd.read_excel(turbine_info_path)

    turbine_tag = infer_turbine_tag(dataname)
    project_number = str(project_number).strip()

    candidate_cols = ["EAM机位号", "业主运行号", "CMS机位号"]
    for col in candidate_cols + ["项目编号", "型号"]:
        if col in turbine_df.columns:
            turbine_df[col] = turbine_df[col].astype(str).str.strip()

    project_match = turbine_df["项目编号"] == project_number
    turbine_tag_norm = normalize_turbine_tag(turbine_tag)
    turbine_match = False
    for col in candidate_cols:
        if col in turbine_df.columns:
            turbine_match = turbine_match | (
                turbine_df[col].map(normalize_turbine_tag) == turbine_tag_norm
            )

    matched_rows = turbine_df.loc[project_match & turbine_match]
    if matched_rows.empty:
        raise ValueError(
            f"the_turbine_info21.xlsx 中未找到项目 {project_number}、机位 {turbine_tag} 的配置"
        )

    model_name = matched_rows.iloc[0]["型号"]
    if not model_name or str(model_name).lower() == "nan":
        raise ValueError(f"项目 {project_number}、机位 {turbine_tag} 的机型为空")

    jixing_df["机型名称"] = jixing_df["机型名称"].astype(str).str.strip()
    jixing_rows = jixing_df.loc[jixing_df["机型名称"] == str(model_name).strip()]
    if jixing_rows.empty:
        raise ValueError(f"test_jixing.xlsx 中未找到机型: {model_name}")

    jixing_row = jixing_rows.iloc[0]
    jixing_type = str(jixing_row["机型类型"]).strip()
    if jixing_type != "二级行星半直驱":
        raise ValueError(f"当前仅支持机型类型: 二级行星半直驱，实际为: {jixing_type}")

    row_text = " ".join(str(v).strip() for v in jixing_row.tolist() if pd.notna(v))
    keywords = [
        "主轴承_",
        "一级行星_",
        "二级行星_",
        "一级太阳轮轴承_",
        "二级太阳轮轴承_",
        "一级行星架轴承_",
        "二级行星架轴承_",
        "一级行星轮轴承_",
        "二级行星轮轴承_",
        "发电机驱动端轴承1_",
        "发电机驱动端轴承2_",
        "发电机非驱动端轴承_",
    ]
    extracted = [extract_keyword_value(row_text, keyword) for keyword in keywords]

    bearing_df["型号"] = bearing_df["型号"].astype(str).str.strip()

    generbearing = BearingParams()
    generbearing.model = [extracted[9], extracted[10], extracted[11]]
    generbearing.parameter = []

    for model in generbearing.model:
        if not model:
            generbearing.parameter.append(
                {"BSF": 0.0, "FTF": 0.0, "BPFO": 0.0, "BPFI": 0.0}
            )
            continue

        matched = bearing_df["型号"] == model
        if not matched.any():
            raise ValueError(f"test_bearing.xlsx 中未找到轴承型号: {model}")

        row = bearing_df.loc[matched].iloc[0]
        generbearing.parameter.append(
            {
                "BSF": float(row["滚子"]),
                "FTF": float(row["保持架"]),
                "BPFO": float(row["外环"]),
                "BPFI": float(row["内环"]),
            }
        )

    faltype = build_faltype_from_generbearing(generbearing)

    log_progress(
        f"配置解析完成: 项目={project_number}, 机位={turbine_tag}, 机型={model_name}, 轴承={generbearing.model}"
    )
    return faltype, generbearing


def build_faltype_from_generbearing(generbearing: BearingParams) -> dict[str, int]:
    faltype: dict[str, int] = {}
    fault_idx = 1
    for model in generbearing.model:
        if not model:
            continue
        faltype[f"WaiHuan{fault_idx}"] = (fault_idx - 1) * 4 + 1
        faltype[f"NeiHuan{fault_idx}"] = (fault_idx - 1) * 4 + 2
        faltype[f"GunDan{fault_idx}"] = (fault_idx - 1) * 4 + 3
        faltype[f"BaoWai{fault_idx}"] = (fault_idx - 1) * 4 + 4
        fault_idx += 1
    return faltype


def build_sql_query(
    dataname: str,
    start_time: str,
    end_time: str,
    rpm_min: float,
    rpm_max: float,
) -> str:
    return (
        'SELECT DATE_FORMAT(date_time, "%Y-%m-%dT%H:%i:%S") AS formatted_date, '
        f"speed, rms_val_acc, std_val_acc, src_path FROM `{dataname}` "
        f"WHERE date_time >= '{start_time}' "
        f"AND date_time <= '{end_time}' "
        f"AND speed >= {rpm_min:g} "
        f"AND speed <= {rpm_max:g} "
        "ORDER BY date_time"
    )


def _run_db_query(db_connector: Any, sql_query: str) -> list[list[Any]]:
    if hasattr(db_connector, "get_data"):
        rows = db_connector.get_data(sql_query)
    elif hasattr(db_connector, "DB_GetData"):
        rows, _ = db_connector.DB_GetData(sql_query, "cell")
    else:
        raise TypeError("db_connector 必须提供 get_data(sql) 或 DB_GetData(sql, 'cell')")

    if rows is None:
        return []
    if rows == []:
        return []
    if isinstance(rows, list) and len(rows) == 1 and rows[0] == "No Data":
        return []
    return rows


def query_db_data(db_connector: Any, sql_query: str) -> QueryData:
    log_progress("开始执行数据库查询")
    rows = _run_db_query(db_connector, sql_query)
    if not rows:
        raise ValueError("数据库检索无数据")

    time_data: list[str] = []
    rpm_data: list[float] = []
    src_path: list[str] = []

    for row in rows:
        if len(row) < 3:
            raise ValueError(f"数据库返回列数不足，至少需要 3 列: {row}")
        time_data.append(str(row[0]))
        rpm_data.append(float(row[1]))
        src_col = row[4] if len(row) >= 5 else row[2]
        src_path.append(str(src_col))

    log_progress(f"数据库查询完成: 共 {len(src_path)} 条记录")
    return QueryData(time_data=time_data, rpm_data=rpm_data, src_path=src_path)


def run_fault_fft_batch(
    src_path: Sequence[str | Path],
    rpm_data: Sequence[float],
    time_data: Sequence[str] | None,
    fs: float,
    faltype: dict[str, int],
    generbearing: BearingParams,
    reader: Callable[[str | Path], np.ndarray] | None = None,
    check_faunum: int = 9,
    use_all_records: bool = True,
    debug_output_dir: str | Path | None = None,
    poles_num: float = 0.0,
    search_band: float = 0.04,
) -> FaultFFTBatchResult:
    if len(src_path) != len(rpm_data):
        raise ValueError("src_path 与 rpm_data 长度不一致")
    if time_data is not None and len(src_path) != len(time_data):
        raise ValueError("src_path 与 time_data 长度不一致")
    if not src_path:
        raise ValueError("src_path 不能为空")
    if not faltype:
        raise ValueError("faltype 不能为空")

    read_numeric = reader or fast_read_numeric
    max_fault_type = max(faltype.values())
    # MATLAB 原逻辑为 1-based: length(src_path)-check_faunum : length(src_path)
    # 转成 Python 0-based 后需要再减 1，否则会从第二条有效记录开始。
    start_idx = 0 if use_all_records else max(len(src_path) - check_faunum - 1, 0)
    total_records = len(src_path) - start_idx
    mode = "全量记录" if use_all_records else f"最近 {len(src_path) - start_idx} 条记录"
    log_progress(f"开始执行 FFT 批处理: {mode}")

    envelope_flags_rows: list[list[bool]] = []
    fft_flags_rows: list[list[bool]] = []
    envelope_values_rows: list[list[float]] = []
    fft_values_rows: list[list[float]] = []
    processed_times: list[str] = []
    maxpoint_en_history: list[dict] = []
    maxpoint_history: list[dict] = []
    mid_history: list[float] = []
    mid_en_history: list[float] = []

    for idx in range(start_idx, len(src_path)):
        current_no = idx - start_idx + 1
        log_progress(
            f"处理文件 {current_no}/{total_records}: path={src_path[idx]}, speed={rpm_data[idx]:g}"
        )
        try:
            raw_data = read_numeric(src_path[idx])
        except Exception as exc:
            log_progress(f"SKIP 文件读取失败: {src_path[idx]} ({exc})")
            continue
        m_bFault_en, m_bFault_zhou, m_ArrMaxPoint_en, m_ArrMaxPoint, mid, mid_en = faultjudge_fft(
            fs=fs,
            shi_data=raw_data,
            speed=float(rpm_data[idx]),
            faltype=faltype,
            generbearing=generbearing,
            debug_output_dir=debug_output_dir,
            debug_prefix=f"fft_record_{current_no:03d}",
            poles_num=poles_num,
            search_band=search_band,
        )

        maxpoint_en_history.append(m_ArrMaxPoint_en)
        maxpoint_history.append(m_ArrMaxPoint)
        mid_history.append(float(mid))
        mid_en_history.append(float(mid_en))

        envelope_flag_row = [False] * max_fault_type
        fft_flag_row = [False] * max_fault_type
        envelope_value_row = [0.0] * max_fault_type
        fft_value_row = [0.0] * max_fault_type

        for fault_type in range(1, max_fault_type + 1):
            envelope_flag_row[fault_type - 1] = bool(m_bFault_en.get(fault_type, False))
            fft_flag_row[fault_type - 1] = bool(m_bFault_zhou.get(fault_type, False))

            en_sum = sum(m_ArrMaxPoint_en[(fault_type, rank)]["value"] for rank in range(1, 6))
            raw_sum = sum(m_ArrMaxPoint[(fault_type, rank)]["value"] for rank in range(1, 6))
            envelope_value_row[fault_type - 1] = en_sum / mid_en if mid_en else 0.0
            fft_value_row[fault_type - 1] = raw_sum / mid if mid else 0.0

        envelope_flags_rows.append(envelope_flag_row)
        fft_flags_rows.append(fft_flag_row)
        envelope_values_rows.append(envelope_value_row)
        fft_values_rows.append(fft_value_row)
        if time_data is not None:
            processed_times.append(str(time_data[idx]))

    envelope_flags = np.asarray(envelope_flags_rows, dtype=bool)
    fft_flags = np.asarray(fft_flags_rows, dtype=bool)
    envelope_values = np.asarray(envelope_values_rows, dtype=float)
    fft_values = np.asarray(fft_values_rows, dtype=float)

    if envelope_flags.size == 0:
        raise ValueError("没有成功处理任何 FFT 文件，请检查 src_path 是否可访问")

    log_progress("FFT 批处理完成")
    return FaultFFTBatchResult(
        maxpoint_en_history=maxpoint_en_history,
        maxpoint_history=maxpoint_history,
        mid=np.asarray(mid_history, dtype=float),
        mid_en=np.asarray(mid_en_history, dtype=float),
        timedatanow=processed_times,
        envelope_flags=envelope_flags,
        fft_flags=fft_flags,
        envelope_values=envelope_values,
        fft_values=fft_values,
    )


def faultjudge_fft_touse(
    time_data: Sequence[str],
    src_path: Sequence[str | Path],
    rpm_data: Sequence[float],
    fs: float,
    generbearing: BearingParams,
    poles_num: float,
    search_band: float,
    reader: Callable[[str | Path], np.ndarray] | None = None,
    debug_output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, list[str]]:
    """
    Python 对应 MATLAB:
    [MaxPoint_en_history, MaxPoint_history, mid, mid_en, timedatanow] =
        faultjudge_fft_touse(timeData, src_path, rpmdata, fs, generbearing, poles_num, search_band)
    """
    faltype = build_faltype_from_generbearing(generbearing)
    batch_result = run_fault_fft_batch(
        src_path=src_path,
        rpm_data=rpm_data,
        time_data=time_data,
        fs=fs,
        faltype=faltype,
        generbearing=generbearing,
        reader=reader,
        use_all_records=True,
        debug_output_dir=debug_output_dir,
        poles_num=poles_num,
        search_band=search_band,
    )
    return (
        batch_result.maxpoint_en_history,
        batch_result.maxpoint_history,
        batch_result.mid,
        batch_result.mid_en,
        batch_result.timedatanow,
    )


def summarize_fft_batch(
    envelope_flags: np.ndarray,
    fft_flags: np.ndarray,
    envelope_values: np.ndarray,
    fft_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    putout_envelop_fau = envelope_flags.sum(axis=0) > 8
    putout_fft_fau = fft_flags.sum(axis=0) > 8
    putout_envelop_value = (envelope_values > 1000).sum(axis=0) > 7
    putout_fft_value = (fft_values > 1000).sum(axis=0) > 7
    return (
        putout_envelop_fau,
        putout_fft_fau,
        putout_envelop_value,
        putout_fft_value,
    )


def _ordered_fault_labels(faltype: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(faltype.items(), key=lambda item: item[1])]


def build_named_fault_outputs(result: FaultFFTRunResult) -> dict[str, dict[str, bool]]:
    labels = _ordered_fault_labels(result.faltype)
    return {
        label: {
            "putout_envelop_fau": bool(result.putout_envelop_fau[idx]),
            "putout_fft_fau": bool(result.putout_fft_fau[idx]),
            "putout_envelop_value": bool(result.putout_envelop_value[idx]),
            "putout_fft_value": bool(result.putout_fft_value[idx]),
        }
        for idx, label in enumerate(labels)
    }


def print_named_fault_outputs(result: FaultFFTRunResult) -> None:
    named_outputs = build_named_fault_outputs(result)
    print("fault_outputs_by_name:")
    for fault_name, values in named_outputs.items():
        print(
            f"  {fault_name}: "
            f"envelop_fau={int(values['putout_envelop_fau'])}, "
            f"fft_fau={int(values['putout_fft_fau'])}, "
            f"envelop_value={int(values['putout_envelop_value'])}, "
            f"fft_value={int(values['putout_fft_value'])}"
        )


def serialise_maxpoint_history(history: Sequence[dict], timedatanow: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for record_idx, points in enumerate(history):
        rows = []
        for fault_type, rank in sorted(points):
            point = points[(fault_type, rank)]
            rows.append(
                {
                    "fault_type": int(fault_type),
                    "rank": int(rank),
                    "number": int(point.get("number", 1)),
                    "value": float(point.get("value", 0.0)),
                }
            )
        output.append(
            {
                "record": record_idx + 1,
                "time": timedatanow[record_idx] if record_idx < len(timedatanow) else "",
                "points": rows,
            }
        )
    return output


def save_maxpoint_history_json(
    history: Sequence[dict],
    timedatanow: Sequence[str],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = serialise_maxpoint_history(history, timedatanow)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def save_maxpoint_history_csv(
    history: Sequence[dict],
    timedatanow: Sequence[str],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write("record,time,fault_type,rank,number,value\n")
        for record_idx, points in enumerate(history):
            record_no = record_idx + 1
            record_time = timedatanow[record_idx] if record_idx < len(timedatanow) else ""
            for fault_type, rank in sorted(points):
                point = points[(fault_type, rank)]
                file.write(
                    f"{record_no},"
                    f"{record_time},"
                    f"{int(fault_type)},"
                    f"{int(rank)},"
                    f"{int(point.get('number', 1))},"
                    f"{float(point.get('value', 0.0)):.12g}\n"
                )
    return output_path


def save_plot_data_points(result: FaultFFTRunResult, path: str | Path) -> Path:
    labels = _ordered_fault_labels(result.faltype)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time"]
    for label in labels:
        header.extend(
            [
                f"{label}_envelop_fau",
                f"{label}_fft_fau",
                f"{label}_envelop_value",
                f"{label}_fft_value",
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(",".join(header) + "\n")
        row_count = result.envelope_flags.shape[0]
        for row_idx in range(row_count):
            time_value = (
                result.plot_times[row_idx]
                if row_idx < len(result.plot_times)
                else str(row_idx)
            )
            row_values = [time_value]
            for col_idx in range(len(labels)):
                row_values.extend(
                    [
                        str(int(result.envelope_flags[row_idx, col_idx])),
                        str(int(result.fft_flags[row_idx, col_idx])),
                        f"{result.envelope_values[row_idx, col_idx]:.12g}",
                        f"{result.fft_values[row_idx, col_idx]:.12g}",
                    ]
                )
            file.write(",".join(row_values) + "\n")
    return output_path


def plot_fault_fft_result(
    result: FaultFFTRunResult,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[Path]:
    import matplotlib.pyplot as plt

    labels = _ordered_fault_labels(result.faltype)
    plot_times = pd.to_datetime(result.plot_times)
    output_dir = Path(save_dir) if save_dir is not None else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    figures = [
        (
            "fft_envelope_overview",
            result.envelope_flags.astype(int),
            result.envelope_values,
            "Envelope Fault Flags",
            "Envelope Energy Ratio",
        ),
        (
            "fft_spectrum_overview",
            result.fft_flags.astype(int),
            result.fft_values,
            "Spectrum Fault Flags",
            "Spectrum Energy Ratio",
        ),
    ]

    for stem, flags, values, title1, title2 in figures:
        log_progress(f"开始绘图并保存: {stem}")
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for idx, label in enumerate(labels):
            axes[0].plot(plot_times, flags[:, idx], linewidth=1.5, label=label)
            axes[1].plot(plot_times, values[:, idx], linewidth=1.5, label=label)

        axes[0].set_title(title1)
        axes[0].set_ylabel("Fault Flag")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left", ncol=2, fontsize=9)

        axes[1].set_title(title2)
        axes[1].set_ylabel("Ratio")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left", ncol=2, fontsize=9)

        fig.autofmt_xdate()
        fig.tight_layout()

        save_path = output_dir / f"{stem}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        saved_paths.append(save_path)
        log_progress(f"图片已保存: {save_path}")

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return saved_paths


def run_fault_fft_from_db(
    db_connector: Any,
    project_number: str,
    dataname: str,
    start_time: str,
    end_time: str,
    fs: float,
    rpm_min: float,
    rpm_max: float,
    config_dir: str | Path | None = None,
    reader: Callable[[str | Path], np.ndarray] | None = None,
    check_faunum: int = 9,
    use_all_records: bool = True,
    debug_output_dir: str | Path | None = None,
    poles_num: float = 0.0,
    search_band: float = 0.04,
) -> FaultFFTRunResult:
    """
    对应 MATLAB 外层调用流程：
    1. 组 SQL
    2. 查数据库得到 time/speed/src_path
    3. 读三张配置表得到 faltype/generbearing
    4. 循环调用 faultjudge_fft
    5. 汇总故障输出
    """
    log_progress("准备构建 SQL")
    sql_query = build_sql_query(
        dataname=dataname,
        start_time=start_time,
        end_time=end_time,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
    )
    log_progress(f"SQL 已生成: {sql_query}")
    query_data = query_db_data(db_connector, sql_query)
    faltype, generbearing = readsome(project_number, dataname, config_dir)
    batch_result = run_fault_fft_batch(
        src_path=query_data.src_path,
        rpm_data=query_data.rpm_data,
        time_data=query_data.time_data,
        fs=fs,
        faltype=faltype,
        generbearing=generbearing,
        reader=reader,
        check_faunum=check_faunum,
        use_all_records=use_all_records,
        debug_output_dir=debug_output_dir,
        poles_num=poles_num,
        search_band=search_band,
    )
    (
        putout_envelop_fau,
        putout_fft_fau,
        putout_envelop_value,
        putout_fft_value,
    ) = summarize_fft_batch(
        envelope_flags=batch_result.envelope_flags,
        fft_flags=batch_result.fft_flags,
        envelope_values=batch_result.envelope_values,
        fft_values=batch_result.fft_values,
    )
    log_progress("FFT 汇总完成")
    plot_times = batch_result.timedatanow

    return FaultFFTRunResult(
        project_number=project_number,
        dataname=dataname,
        sql_query=sql_query,
        query_data=query_data,
        faltype=faltype,
        generbearing=generbearing,
        maxpoint_en_history=batch_result.maxpoint_en_history,
        maxpoint_history=batch_result.maxpoint_history,
        mid=batch_result.mid,
        mid_en=batch_result.mid_en,
        timedatanow=batch_result.timedatanow,
        envelope_flags=batch_result.envelope_flags,
        fft_flags=batch_result.fft_flags,
        envelope_values=batch_result.envelope_values,
        fft_values=batch_result.fft_values,
        putout_envelop_fau=putout_envelop_fau,
        putout_fft_fau=putout_fft_fau,
        putout_envelop_value=putout_envelop_value,
        putout_fft_value=putout_fft_value,
        plot_times=plot_times,
    )


class PreviewDatabaseConnector:
    """
    命令行演示用。
    不实际连库，只把用户提供的 SQL 结果文本解析成 rows。
    每行格式：
    2026-06-10T00:00:00,1500,D:\\path\\a.csv
    """

    def __init__(self, rows: Sequence[Sequence[Any]]):
        self._rows = [list(row) for row in rows]

    def get_data(self, sql: str) -> list[list[Any]]:
        del sql
        return self._rows


def parse_preview_rows(raw: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise ValueError(f"preview-data 每行必须是 3 列: {line}")
        rows.append([parts[0], float(parts[1]), parts[2]])
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 MATLAB 调用流程执行 FFT 故障诊断")
    parser.add_argument("--project-number", help="项目编号，例如 S1-20180011")
    parser.add_argument("--dataname", help="测点表名，例如 23#_generator_发电机12点径向_25600.0_0.0_0.0")
    parser.add_argument("--start-time", help="开始时间，例如 2022-06-10 00:00:00")
    parser.add_argument("--end-time", help="结束时间，例如 2026-06-10 00:00:00")
    parser.add_argument("--fs", type=float, help="采样频率，例如 25600")
    parser.add_argument("--rpm-min", type=float, help="最小转速，例如 1.72")
    parser.add_argument("--rpm-max", type=float, help="最大转速，例如 21543")
    parser.add_argument("--config-dir", help="配置目录，默认读取配置文件中的值")
    parser.add_argument("--check-faunum", type=int, help="回看样本窗口，默认读取配置文件中的值")
    parser.add_argument("--use-all-records", action="store_true", help="对查询时间范围内所有记录都执行 FFT，并画完整时间范围")
    parser.add_argument("--poles-num", type=float, help="半直驱机型的发电机极个数，常规双馈为 0")
    parser.add_argument("--search-band", type=float, help="峰值搜索宽度系数，默认读取配置文件中的值")
    parser.add_argument("--db-host", help="数据库地址")
    parser.add_argument("--db-port", type=int, help="数据库端口")
    parser.add_argument("--db-user", help="数据库用户名")
    parser.add_argument("--db-password", help="数据库密码")
    parser.add_argument("--db-database", help="数据库名")
    parser.add_argument("--output-dir", help="图像输出目录，默认 D:\\CMS\\output")
    parser.add_argument("--no-plot", action="store_true", help="只保存图，不弹出图窗")
    parser.add_argument(
        "--preview-data",
        help="可选。用文本模拟数据库返回，每行格式: 时间,转速,文件路径",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = DEFAULT_FAULT_FFT_CONFIG
    project_number = args.project_number or cfg.query.project_number
    dataname = args.dataname or cfg.query.dataname
    start_time = args.start_time or cfg.query.start_time
    end_time = args.end_time or cfg.query.end_time
    fs = args.fs if args.fs is not None else cfg.query.fs
    rpm_min = args.rpm_min if args.rpm_min is not None else cfg.query.rpm_min
    rpm_max = args.rpm_max if args.rpm_max is not None else cfg.query.rpm_max
    config_dir = args.config_dir or str(cfg.paths.config_dir)
    check_faunum = args.check_faunum if args.check_faunum is not None else cfg.query.check_faunum
    use_all_records = args.use_all_records or cfg.query.use_all_records
    poles_num = args.poles_num if args.poles_num is not None else cfg.query.poles_num
    search_band = args.search_band if args.search_band is not None else cfg.query.search_band
    preview_data = args.preview_data if args.preview_data is not None else cfg.preview_data

    db_host = args.db_host or cfg.db.host
    db_port = args.db_port if args.db_port is not None else cfg.db.port
    db_user = args.db_user or cfg.db.user
    db_password = args.db_password or cfg.db.password
    db_database = args.db_database or cfg.db.database
    output_dir = args.output_dir or str(PROJECT_ROOT / "output")
    output_path = Path(output_dir)
    debug_output_dir = output_path / "fft_debug"

    log_progress("入口参数已解析")
    log_progress(
        f"运行配置: project={project_number}, dataname={dataname}, start={start_time}, end={end_time}, fs={fs}, rpm=[{rpm_min:g}, {rpm_max:g}], use_all_records={use_all_records}, poles_num={poles_num:g}, search_band={search_band:g}"
    )

    if preview_data:
        log_progress("使用 preview_data 模拟数据库结果")
        preview_rows = parse_preview_rows(preview_data)
        db_connector = PreviewDatabaseConnector(preview_rows)
        should_close = False
    else:
        log_progress(f"连接数据库: host={db_host}, port={db_port}, database={db_database}, user={db_user}")
        db_connector = MySQLConnector(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_database,
        )
        should_close = True

    try:
        result = run_fault_fft_from_db(
            db_connector=db_connector,
            project_number=project_number,
            dataname=dataname,
            start_time=start_time,
            end_time=end_time,
            fs=fs,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            config_dir=config_dir,
            check_faunum=check_faunum,
            use_all_records=use_all_records,
            debug_output_dir=debug_output_dir,
            poles_num=poles_num,
            search_band=search_band,
        )
    finally:
        if should_close and hasattr(db_connector, "close"):
            db_connector.close()
            log_progress("数据库连接已关闭")

    output_path.mkdir(parents=True, exist_ok=True)
    maxpoint_en_json_path = save_maxpoint_history_json(
        result.maxpoint_en_history,
        result.timedatanow,
        output_path / "maxpoint_en_history.json",
    )
    maxpoint_en_csv_path = save_maxpoint_history_csv(
        result.maxpoint_en_history,
        result.timedatanow,
        output_path / "maxpoint_en_history.csv",
    )
    maxpoint_json_path = save_maxpoint_history_json(
        result.maxpoint_history,
        result.timedatanow,
        output_path / "maxpoint_history.json",
    )
    maxpoint_csv_path = save_maxpoint_history_csv(
        result.maxpoint_history,
        result.timedatanow,
        output_path / "maxpoint_history.csv",
    )
    summary_path = output_path / "result_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"sql_query: {result.sql_query}\n")
        file.write(f"rows: {len(result.query_data.src_path)}\n")
        file.write(f"faltype: {result.faltype}\n")
        file.write(f"generbearing.model: {result.generbearing.model}\n")
        file.write(f"maxpoint_en_history_json: {maxpoint_en_json_path}\n")
        file.write(f"maxpoint_en_history_csv: {maxpoint_en_csv_path}\n")
        file.write(f"maxpoint_history_json: {maxpoint_json_path}\n")
        file.write(f"maxpoint_history_csv: {maxpoint_csv_path}\n")
        file.write(f"maxpoint_en_history: {serialise_maxpoint_history(result.maxpoint_en_history, result.timedatanow)}\n")
        file.write(f"maxpoint_history: {serialise_maxpoint_history(result.maxpoint_history, result.timedatanow)}\n")
        file.write(f"mid: {result.mid.tolist()}\n")
        file.write(f"mid_en: {result.mid_en.tolist()}\n")
        file.write(f"timedatanow: {result.timedatanow}\n")
        file.write(f"putout_envelop_fau: {result.putout_envelop_fau.astype(int).tolist()}\n")
        file.write(f"putout_fft_fau: {result.putout_fft_fau.astype(int).tolist()}\n")
        file.write(f"putout_envelop_value: {result.putout_envelop_value.astype(int).tolist()}\n")
        file.write(f"putout_fft_value: {result.putout_fft_value.astype(int).tolist()}\n")
    plot_data_path = save_plot_data_points(result, output_path / "plot_data_points.csv")
    saved_paths = plot_fault_fft_result(
        result=result,
        save_dir=output_dir,
        show=not args.no_plot,
    )
    saved_info_path = output_path / "saved_outputs.txt"
    with saved_info_path.open("w", encoding="utf-8") as file:
        file.write(f"summary: {summary_path}\n")
        file.write(f"plot_data_points: {plot_data_path}\n")
        file.write(f"debug_output_dir: {debug_output_dir}\n")
        file.write(f"saved_plots: {[str(path) for path in saved_paths]}\n")
    log_progress("全部处理完成")


if __name__ == "__main__":
    main()
