from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def envelope_signal(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 FFT 形式的 Hilbert 变换计算包络。
    返回包络和解析信号，接口兼容 fault_fft 的调用方式。
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = x.size
    spectrum = np.fft.fft(x)

    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0

    analytic_signal = np.fft.ifft(spectrum * h)
    envelope = np.abs(analytic_signal)
    return envelope, analytic_signal


def hann_window(length: int) -> np.ndarray:
    return np.hanning(length)


def compute_fft_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    length = signal.size
    fft_result = np.fft.fft(signal)
    p2 = np.abs(fft_result / length)
    p1 = p2[: length // 2 + 1].copy()
    if p1.size > 2:
        p1[1:-1] *= 2.0

    frequency = fs * np.arange(0, length // 2 + 1) / length
    p1_rms = p1 / np.sqrt(2.0)
    return p1_rms, frequency


def hampel_filter(
    values: np.ndarray,
    window_size: int = 10,
    n_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hampel 滤波。
    返回滤波后的数组和离群点布尔掩码。
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    filtered = x.copy()
    outlier_mask = np.zeros_like(x, dtype=bool)

    if x.size == 0:
        return filtered, outlier_mask

    k = max(int(window_size), 1)
    scale = 1.4826

    for i in range(x.size):
        start = max(0, i - k)
        end = min(x.size, i + k + 1)
        window = x[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        threshold = n_sigma * scale * mad
        if mad > 0 and abs(x[i] - median) > threshold:
            filtered[i] = median
            outlier_mask[i] = True

    return filtered, outlier_mask


def resample_to_fixed_interval(
    time_data,
    values: np.ndarray,
    interval_hours: float = 2.0,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    按固定小时间隔重采样，使用线性插值，对齐 MATLAB 中的 interp1 逻辑。
    """
    time_index = pd.to_datetime(time_data)
    y = np.asarray(values, dtype=float).reshape(-1)

    if len(time_index) != len(y):
        raise ValueError("time_data 与 values 长度不一致")
    if len(time_index) == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=float)

    start = time_index[0]
    end = time_index[-1]
    custom_times = pd.date_range(start=start, end=end, freq=pd.to_timedelta(interval_hours, unit="h"))
    if len(custom_times) == 0 or custom_times[-1] != end:
        custom_times = custom_times.union(pd.DatetimeIndex([end]))

    x_original = time_index.view("int64").astype(float)
    x_target = custom_times.view("int64").astype(float)
    interp_values = np.interp(x_target, x_original, y)
    return custom_times, interp_values


def moving_percentile(
    values: np.ndarray,
    window_len: int = 200,
    pct: float = 95.0,
) -> np.ndarray:
    """
    滑动百分位数，边界窗口自动收缩。
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    result = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return result

    def matlab_prctile(window: np.ndarray, percentile: float) -> float:
        sorted_values = np.sort(window[~np.isnan(window)])
        n = sorted_values.size
        if n == 0:
            return float("nan")
        if n == 1:
            return float(sorted_values[0])

        position = n * float(percentile) / 100.0 + 0.5
        if position <= 1.0:
            return float(sorted_values[0])
        if position >= n:
            return float(sorted_values[-1])

        lower_rank = int(np.floor(position))
        fraction = position - lower_rank
        lower_value = sorted_values[lower_rank - 1]
        upper_value = sorted_values[lower_rank]
        return float(lower_value + fraction * (upper_value - lower_value))

    half_window = max(int(window_len) // 2, 0)
    for i in range(x.size):
        start = max(0, i - half_window)
        end = min(x.size, i + half_window + 1)
        result[i] = matlab_prctile(x[start:end], pct)

    return result


def moving_max(values: np.ndarray, window: int = 200) -> np.ndarray:
    """
    滑动最大值，等价 MATLAB movmax 的居中窗口行为。
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    result = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return result

    half_window = max(int(window) // 2, 0)
    for i in range(x.size):
        start = max(0, i - half_window)
        end = min(x.size, i + half_window + 1)
        result[i] = np.max(x[start:end])

    return result


def days_from_origin(time_values, origin) -> np.ndarray:
    """
    将时间序列转换成相对 origin 的天数。
    """
    times = pd.to_datetime(time_values)
    origin_ts = pd.to_datetime(origin)
    delta = times - origin_ts
    return np.asarray(delta.total_seconds() / 86400.0, dtype=float)
