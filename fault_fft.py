# =============================================================
#  FFT 频谱故障诊断模块
#  等效 MATLAB: faultjudge_fft
# =============================================================
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
from utils.signal_processing import compute_fft_spectrum, envelope_signal, hann_window


def _fault_type_range(faltype: dict) -> range:
    ntype_values = sorted(faltype.values())
    expected_values = list(range(ntype_values[0], ntype_values[-1] + 1))
    if ntype_values != expected_values:
        raise ValueError("faltype 编号必须连续")
    return range(ntype_values[0], ntype_values[-1] + 1)


def matlab_round(value: float) -> int:
    """Match MATLAB round: .5 values round away from zero."""
    value = float(value)
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


def init_mavin_sys(speed: float, faltype: dict) -> Tuple[dict, dict, float]:
    """初始化故障标志、峰值点容器和转频。"""
    ntype_range = _fault_type_range(faltype)
    m_bFault = {ntype: True for ntype in ntype_range}
    m_ArrMaxPoint = {
        (ntype, nrank): {"value": 0.0, "number": 1}
        for ntype in ntype_range
        for nrank in range(1, 6)
    }
    speed_fra = speed / 60.0
    return m_bFault, m_ArrMaxPoint, speed_fra


def compute_fault_fre(
    speed_fra: float,
    faltype: dict,
    resolving: float,
    m_bFault: dict,
    generbearing,
) -> dict:
    """
    计算各故障类型对应的理论基频谱线号。
    MATLAB 原函数接收 m_bFault，但当前逻辑中不参与计算，保留参数仅为接口一致性。
    """
    del m_bFault

    ntype_range = _fault_type_range(faltype)
    m_dblArrBaseFre = {ntype: 0.0 for ntype in ntype_range}
    m_dblArrMapBaseFre = {ntype: 0.0 for ntype in ntype_range}
    base_fre = speed_fra

    bearing_idx = 1
    if len(generbearing.model) != len(generbearing.parameter):
        raise ValueError("generbearing.model 与 generbearing.parameter 长度不一致")

    for model, parameter in zip(generbearing.model, generbearing.parameter):
        if not model:
            continue

        mapping = [
            (f"WaiHuan{bearing_idx}", "BPFO"),
            (f"NeiHuan{bearing_idx}", "BPFI"),
            (f"GunDan{bearing_idx}", "BSF"),
            (f"BaoWai{bearing_idx}", "FTF"),
        ]
        for fault_name, coef_name in mapping:
            if fault_name in faltype:
                m_dblArrBaseFre[faltype[fault_name]] = parameter[coef_name] * base_fre

        bearing_idx += 1

    for ntype in ntype_range:
        m_dblArrMapBaseFre[ntype] = (
            m_dblArrBaseFre[ntype] / resolving if resolving > 0 else 0.0
        )

    return m_dblArrMapBaseFre


def search_max_point_en(
    faltype: dict,
    m_dblArrMapBaseFre: dict,
    fft_result_len: int,
    m_ArrMaxPoint: dict,
    spectrum: np.ndarray,
    search_band: float = 0.04,
    debug_pu_num_path: str | Path | None = None,
    debug_range_path: str | Path | None = None,
    debug_range_name: str = "m_dblArrRange",
) -> dict:
    """
    搜索每类故障 1~5 阶的极值点。
    保持 MATLAB 的 1-based 频线号语义，内部访问 numpy 时转换成 0-based。
    """
    if not np.isfinite(search_band) or search_band < 0:
        raise ValueError("search_band 必须是非负有限数")

    ntype_range = _fault_type_range(faltype)
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)

    m_nArrPuNum = {}
    m_nArrSearchWidth = {}
    pu_num_debug_rows = []
    range_debug_rows = []

    for nrank in range(1, 6):
        for ntype in ntype_range:
            map_base = nrank * m_dblArrMapBaseFre[ntype]
            rounded_map_base = matlab_round(map_base)
            pu_num = rounded_map_base + 1
            raw_search_width = matlab_round(0.5 + pu_num * search_band)
            if pu_num >= fft_result_len:
                m_nArrPuNum[(ntype, nrank)] = 0
                pu_num_debug_rows.append(
                    {
                        "fault_type": ntype,
                        "rank": nrank,
                        "map_base": map_base,
                        "rounded_map_base": rounded_map_base,
                        "pu_num": pu_num,
                        "effective_pu_num": 0,
                        "fft_result_len": fft_result_len,
                        "raw_search_width": raw_search_width,
                        "search_width": 0,
                        "skipped": 1,
                    }
                )
                continue

            m_nArrPuNum[(ntype, nrank)] = pu_num
            search_width = max(1, min(32, raw_search_width))
            m_nArrSearchWidth[(ntype, nrank)] = search_width
            pu_num_debug_rows.append(
                {
                    "fault_type": ntype,
                    "rank": nrank,
                    "map_base": map_base,
                    "rounded_map_base": rounded_map_base,
                    "pu_num": pu_num,
                    "effective_pu_num": pu_num,
                    "fft_result_len": fft_result_len,
                    "raw_search_width": raw_search_width,
                    "search_width": search_width,
                    "skipped": 0,
                }
            )

    if debug_pu_num_path is not None:
        save_pu_num_debug(debug_pu_num_path, pu_num_debug_rows, faltype)

    for ntype in ntype_range:
        for nrank in range(1, 6):
            pu_num = m_nArrPuNum.get((ntype, nrank), 0)
            if pu_num == 0:
                continue

            search_width = m_nArrSearchWidth[(ntype, nrank)]
            start_pos = max(2, pu_num - search_width)
            end_pos = min(fft_result_len - 1, pu_num + search_width)

            dbl_max_val = 0.0
            n_max_pos = 1
            for matlab_idx in range(start_pos, end_pos + 1):
                value = spectrum[matlab_idx - 1]
                range_debug_rows.append(
                    {
                        "fault_type": ntype,
                        "rank": nrank,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "matlab_index": matlab_idx,
                        "python_index": matlab_idx - 1,
                        "value": float(value),
                    }
                )
                if value > dbl_max_val:
                    dbl_max_val = float(value)
                    n_max_pos = matlab_idx

            m_ArrMaxPoint[(ntype, nrank)]["value"] = dbl_max_val
            m_ArrMaxPoint[(ntype, nrank)]["number"] = n_max_pos

    if debug_range_path is not None:
        save_range_debug(debug_range_path, range_debug_rows, faltype, debug_range_name)

    return m_ArrMaxPoint


def save_range_debug(
    path: str | Path,
    rows: list[dict],
    faltype: dict,
    value_name: str,
) -> None:
    fault_name_by_type = {value: key for key, value in faltype.items()}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "fault_type,fault_name,rank,start_pos,end_pos,"
            "matlab_index,python_index,expression,value\n"
        )
        for row in rows:
            fault_type = row["fault_type"]
            fault_name = fault_name_by_type.get(fault_type, f"type_{fault_type}")
            matlab_index = row["matlab_index"]
            file.write(
                f"{fault_type},"
                f"{fault_name},"
                f"{row['rank']},"
                f"{row['start_pos']},"
                f"{row['end_pos']},"
                f"{matlab_index},"
                f"{row['python_index']},"
                f"{value_name}({matlab_index}),"
                f"{row['value']:.12g}\n"
            )


def save_pu_num_debug(path: str | Path, rows: list[dict], faltype: dict) -> None:
    fault_name_by_type = {value: key for key, value in faltype.items()}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "fault_type,fault_name,rank,map_base,rounded_map_base,"
            "pu_num,effective_pu_num,fft_result_len,raw_search_width,"
            "search_width,skipped\n"
        )
        for row in rows:
            fault_type = row["fault_type"]
            fault_name = fault_name_by_type.get(fault_type, f"type_{fault_type}")
            file.write(
                f"{fault_type},"
                f"{fault_name},"
                f"{row['rank']},"
                f"{row['map_base']:.12g},"
                f"{row['rounded_map_base']},"
                f"{row['pu_num']},"
                f"{row['effective_pu_num']},"
                f"{row['fft_result_len']},"
                f"{row['raw_search_width']},"
                f"{row['search_width']},"
                f"{row['skipped']}\n"
            )


def analyse_superposition(
    faltype: dict,
    m_ArrMaxPoint: dict,
    m_bFault: dict,
    m_dblArrMapBaseFre: dict,
    sf_map: float = 0.0,
    debug_path: str | Path | None = None,
) -> Tuple[dict, dict]:
    """五阶完全重合去重，并排除更接近同步频率的干扰谱线。"""
    ntype_range = list(_fault_type_range(faltype))
    debug_rows = []

    for nTypeA in ntype_range:
        if not m_bFault.get(nTypeA, False):
            continue

        for nTypeB in ntype_range:
            if nTypeA == nTypeB or not m_bFault.get(nTypeB, False):
                continue

            n_count = 0
            n_distance_a = 0.0
            n_distance_b = 0.0

            for nrank in range(1, 6):
                point_a = m_ArrMaxPoint[(nTypeA, nrank)]["number"]
                point_b = m_ArrMaxPoint[(nTypeB, nrank)]["number"]
                if point_a == point_b:
                    n_count += 1
                    n_distance_a += abs(nrank * m_dblArrMapBaseFre[nTypeA] - point_a)
                    n_distance_b += abs(nrank * m_dblArrMapBaseFre[nTypeB] - point_b)

            if n_count == 5:
                loser = nTypeA if n_distance_a > n_distance_b else nTypeB
                debug_rows.append(
                    {
                        "reason": "same_points",
                        "fault_type_a": nTypeA,
                        "fault_type_b": nTypeB,
                        "loser": loser,
                        "n_count": n_count,
                        "distance_a": n_distance_a,
                        "distance_b": n_distance_b,
                        "sf_map": sf_map,
                    }
                )
                m_bFault[loser] = False
                for nrank in range(1, 6):
                    m_ArrMaxPoint[(loser, nrank)]["value"] = 0.0

        if sf_map:
            n_distance_a_sf = 0.0
            n_distance_b_sf = 0.0
            for nrank_sf in range(1, 6):
                point = m_ArrMaxPoint[(nTypeA, nrank_sf)]["number"]
                n_distance_a_sf += abs(
                    nrank_sf * m_dblArrMapBaseFre[nTypeA] - point
                )
                n_distance_b_sf += abs(nrank_sf * sf_map - point)

            if n_distance_a_sf > n_distance_b_sf:
                debug_rows.append(
                    {
                        "reason": "sync_frequency",
                        "fault_type_a": nTypeA,
                        "fault_type_b": 0,
                        "loser": nTypeA,
                        "n_count": 0,
                        "distance_a": n_distance_a_sf,
                        "distance_b": n_distance_b_sf,
                        "sf_map": sf_map,
                    }
                )
                m_bFault[nTypeA] = False
                for nrank in range(1, 6):
                    m_ArrMaxPoint[(nTypeA, nrank)]["value"] = 0.0

    if debug_path is not None:
        save_superposition_debug(debug_path, debug_rows)

    return m_bFault, m_ArrMaxPoint


def save_superposition_debug(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "reason,fault_type_a,fault_type_b,loser,n_count,"
            "distance_a,distance_b,sf_map\n"
        )
        for row in rows:
            file.write(
                f"{row['reason']},"
                f"{row['fault_type_a']},"
                f"{row['fault_type_b']},"
                f"{row['loser']},"
                f"{row['n_count']},"
                f"{row['distance_a']:.12g},"
                f"{row['distance_b']:.12g},"
                f"{row['sf_map']:.12g}\n"
            )


def mavin_alert_en(
    faltype: dict,
    m_ArrMaxPoint: dict,
    m_bFault: dict,
    m_dblRangeAverage: float,
    speed: float,
    m_dblArrRange: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    """
    MATLAB 中的 dB 判据函数。
    当前主流程没有启用，但按截图补齐，便于后续切回该逻辑。
    """
    del m_dblRangeAverage, speed

    ntype_range = _fault_type_range(faltype)
    dblRangeArr = np.zeros((max(ntype_range), 5), dtype=float)

    non_zero = np.asarray(m_dblArrRange, dtype=float)
    non_zero = non_zero[non_zero != 0]
    minrange = float(np.min(non_zero)) if non_zero.size else 1.0

    for ntype in ntype_range:
        if not m_bFault.get(ntype, False):
            m_bFault[ntype] = False
            continue

        m_bFault_point = 0
        for nrank in range(1, 6):
            value = m_ArrMaxPoint[(ntype, nrank)]["value"]
            if value == 0:
                dblRangeArr[ntype - 1, nrank - 1] = 0.0
                m_bFault_point += 1
                continue

            db_value = 20.0 * np.log10(value / minrange)
            dblRangeArr[ntype - 1, nrank - 1] = db_value
            if db_value <= 90:
                m_bFault_point += 1

        if m_bFault_point > 1:
            m_bFault[ntype] = False

    return m_bFault, dblRangeArr


def save_arr_max_points(path: str | Path, points: dict, faltype: dict) -> None:
    fault_name_by_type = {value: key for key, value in faltype.items()}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write("fault_type,fault_name,rank,number,value\n")
        for fault_type in _fault_type_range(faltype):
            fault_name = fault_name_by_type.get(fault_type, f"type_{fault_type}")
            for rank in range(1, 6):
                point = points[(fault_type, rank)]
                file.write(
                    f"{fault_type},"
                    f"{fault_name},"
                    f"{rank},"
                    f"{point['number']},"
                    f"{point['value']:.12g}\n"
                )


def save_spectrum(path: str | Path, spectrum: np.ndarray) -> None:
    values = np.asarray(spectrum, dtype=float).reshape(-1)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write("index,value\n")
        for idx, value in enumerate(values, start=1):
            file.write(f"{idx},{value:.12g}\n")


def save_indexed_values(path: str | Path, name: str, values: np.ndarray) -> None:
    indexed_values = np.asarray(values, dtype=float).reshape(-1)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        for idx, value in enumerate(indexed_values):
            file.write(f"{name}[{idx}]={value:.12g}\n")


def save_signal_values(path: str | Path, values: np.ndarray) -> None:
    signal_values = np.asarray(values, dtype=float).reshape(-1)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write("index,value\n")
        for idx, value in enumerate(signal_values, start=1):
            file.write(f"{idx},{value:.12g}\n")


def faultjudge_fft(
    fs: float,
    shi_data: np.ndarray,
    speed: float,
    faltype: dict,
    generbearing,
    debug_output_dir: str | Path | None = None,
    debug_prefix: str = "fft",
    poles_num: float = 0.0,
    search_band: float = 0.04,
) -> Tuple[dict, dict, dict, dict, float, float]:
    """
    FFT 频谱故障诊断主函数。

    Returns
    -------
    m_bFault_en, m_bFault, m_ArrMaxPoint_en, m_ArrMaxPoint, mid, mid_en
    """
    shi_data = np.asarray(shi_data, dtype=float).reshape(-1)
    if shi_data.size < 2:
        raise ValueError("shi_data 至少需要 2 个采样点")
    if fs <= 0:
        raise ValueError("fs 必须大于 0")
    if not faltype:
        raise ValueError("faltype 不能为空")

    shi_data = shi_data - np.mean(shi_data)
    signal_length = shi_data.size

    up_envelp, _ = envelope_signal(shi_data)
    up_envelp = up_envelp - np.mean(up_envelp)

    window = hann_window(signal_length)
    window_mean = np.mean(window)
    wav_signal = (shi_data * window) / window_mean
    env_data = (up_envelp * window) / window_mean

    p1_rms, _ = compute_fft_spectrum(wav_signal, fs)
    p1_rms_en, _ = compute_fft_spectrum(env_data, fs)

    resolving = fs / signal_length
    sf_map = (speed * poles_num / 120.0) / resolving if poles_num else 0.0
    fft_result_len = len(p1_rms_en)
    m_dblArrRange = p1_rms
    mid = float(np.median(p1_rms))

    m_dblArrRange_en = p1_rms_en
    mid_en = float(np.median(p1_rms_en))

    m_bFault, m_ArrMaxPoint_template, speed_fra = init_mavin_sys(speed, faltype)
    m_dblArrMapBaseFre = compute_fault_fre(
        speed_fra, faltype, resolving, m_bFault, generbearing
    )
    debug_dir = Path(debug_output_dir) if debug_output_dir is not None else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    m_ArrMaxPoint_en = search_max_point_en(
        faltype,
        m_dblArrMapBaseFre,
        fft_result_len,
        deepcopy(m_ArrMaxPoint_template),
        m_dblArrRange_en,
        search_band,
        debug_dir / f"{debug_prefix}_pu_num_en.csv" if debug_dir is not None else None,
        (
            debug_dir / f"{debug_prefix}_m_dblArrRange_en_window.csv"
            if debug_dir is not None
            else None
        ),
        "m_dblArrRange_en",
    )
    if debug_dir is not None:
        save_arr_max_points(
            debug_dir / f"{debug_prefix}_m_ArrMaxPoint_en_before_superposition.csv",
            m_ArrMaxPoint_en,
            faltype,
        )
    m_bFault, m_ArrMaxPoint_en = analyse_superposition(
        faltype,
        m_ArrMaxPoint_en,
        m_bFault,
        m_dblArrMapBaseFre,
        sf_map,
        (
            debug_dir / f"{debug_prefix}_superposition_en.csv"
            if debug_dir is not None
            else None
        ),
    )

    ntype_range = _fault_type_range(faltype)
    m_bFault_zhou = {ntype: bool(m_bFault.get(ntype, False)) for ntype in ntype_range}

    m_ArrMaxPoint = search_max_point_en(
        faltype,
        m_dblArrMapBaseFre,
        fft_result_len,
        deepcopy(m_ArrMaxPoint_template),
        m_dblArrRange,
        search_band,
        debug_dir / f"{debug_prefix}_pu_num.csv" if debug_dir is not None else None,
        (
            debug_dir / f"{debug_prefix}_m_dblArrRange_window.csv"
            if debug_dir is not None
            else None
        ),
        "m_dblArrRange",
    )
    if debug_dir is not None:
        save_arr_max_points(
            debug_dir / f"{debug_prefix}_m_ArrMaxPoint_before_superposition.csv",
            m_ArrMaxPoint,
            faltype,
        )
    m_bFault_zhou, m_ArrMaxPoint = analyse_superposition(
        faltype,
        m_ArrMaxPoint,
        m_bFault_zhou,
        m_dblArrMapBaseFre,
        sf_map,
        (
            debug_dir / f"{debug_prefix}_superposition.csv"
            if debug_dir is not None
            else None
        ),
    )

    if debug_output_dir is not None:
        save_signal_values(debug_dir / f"{debug_prefix}_shi_data.csv", shi_data)
        save_signal_values(debug_dir / f"{debug_prefix}_up_envelp.csv", up_envelp)
        save_spectrum(debug_dir / f"{debug_prefix}_spectrum_en.csv", m_dblArrRange_en)
        save_spectrum(debug_dir / f"{debug_prefix}_spectrum.csv", m_dblArrRange)
        save_indexed_values(
            debug_dir / f"{debug_prefix}_m_dblArrRange_en_indexed.txt",
            "m_dblArrRange_en",
            m_dblArrRange_en,
        )
        save_indexed_values(
            debug_dir / f"{debug_prefix}_m_dblArrRange_indexed.txt",
            "m_dblArrRange",
            m_dblArrRange,
        )
        save_arr_max_points(
            debug_dir / f"{debug_prefix}_m_ArrMaxPoint_en.csv",
            m_ArrMaxPoint_en,
            faltype,
        )
        save_arr_max_points(
            debug_dir / f"{debug_prefix}_m_ArrMaxPoint.csv",
            m_ArrMaxPoint,
            faltype,
        )

    return (
        m_bFault,
        m_bFault_zhou,
        m_ArrMaxPoint_en,
        m_ArrMaxPoint,
        mid,
        mid_en,
    )

