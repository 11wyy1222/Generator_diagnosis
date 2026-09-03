from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def rolling_median(values: np.ndarray, window: int = 31) -> np.ndarray:
    result = np.empty(values.size, dtype=float)
    half = window // 2
    for index in range(values.size):
        result[index] = np.median(values[max(0, index - half): min(values.size, index + half + 1)])
    return result


def draw_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    times: list[datetime],
    values: np.ndarray,
    color: str,
    title: str,
    y_low: float,
    y_high: float,
    detail: bool,
) -> None:
    left, top, right, bottom = bounds
    axis_color, grid_color, muted = "#334155", "#d8e0e8", "#64748b"
    draw.rectangle(bounds, fill="#f8fafc", outline="#cbd5e1", width=2)
    pad_left, pad_right, pad_top, pad_bottom = 95, 28, 58, 68
    x0, x1, y0, y1 = left + pad_left, right - pad_right, top + pad_top, bottom - pad_bottom
    start, end = min(times), max(times)
    seconds = max((end - start).total_seconds(), 1.0)

    def px(time: datetime) -> float:
        return x0 + (time - start).total_seconds() / seconds * (x1 - x0)

    def py(value: float) -> float:
        return y1 - (value - y_low) / max(y_high - y_low, 1e-12) * (y1 - y0)

    draw.text((left + 18, top + 12), title, fill="#172033", font=font(23, True))
    for index in range(5):
        fraction = index / 4
        value = y_low + fraction * (y_high - y_low)
        y = py(value)
        draw.line((x0, y, x1, y), fill=grid_color, width=1)
        label = f"{value:.6f}" if detail else f"{value:.2f}"
        box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((x0 - 10 - (box[2] - box[0]), y - 10), label, fill=axis_color, font=font(17))
    for index in range(5):
        fraction = index / 4
        time = start + (end - start) * fraction
        x = x0 + fraction * (x1 - x0)
        draw.line((x, y0, x, y1), fill=grid_color, width=1)
        label = time.strftime("%Y-%m")
        box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((x - (box[2] - box[0]) / 2, y1 + 15), label, fill=axis_color, font=font(17))
    draw.line((x0, y0, x0, y1), fill=axis_color, width=2)
    draw.line((x0, y1, x1, y1), fill=axis_color, width=2)

    points = [(px(time), py(float(value))) for time, value in zip(times, values)]
    if detail:
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        median = rolling_median(values)
        median_points = [(px(time), py(float(value))) for time, value in zip(times, median)]
        draw.line(median_points, fill=color, width=4, joint="curve")
        minimum_index = int(np.argmin(values))
        x, y = points[minimum_index]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#dc2626", outline="white", width=2)
        label = f"最低 {values[minimum_index]:.6f}"
        label_x = min(max(x + 10, x0 + 5), x1 - 190)
        label_y = min(max(y + 10, y0 + 5), y1 - 30)
        draw.text((label_x, label_y), label, fill="#b91c1c", font=font(18, True))
        draw.text((x1 - 250, y0 + 8), "细线点：单条波形  粗线：31条滚动中位数", fill=muted, font=font(16))
    else:
        draw.line(points, fill=color, width=2, joint="curve")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-sensor probability trends")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="异常概率时间趋势")
    args = parser.parse_args()

    rows = pq.read_table(
        args.parquet,
        columns=["object_id", "sensor_position", "acquisition_time", "abnormal_probability"],
    ).to_pylist()
    rows = [row for row in rows if str(row["object_id"]) == args.object_id]
    if not rows:
        raise ValueError(f"no records found for object {args.object_id}")

    sensors = ["3点", "6点", "9点", "12点"]
    colors = {"3点": "#2563eb", "6点": "#059669", "9点": "#d97706", "12点": "#7c3aed"}
    grouped: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for sensor in sensors:
        selected = sorted(
            (datetime.fromisoformat(str(row["acquisition_time"])), float(row["abnormal_probability"]))
            for row in rows if str(row["sensor_position"]) == sensor
        )
        grouped[sensor] = ([item[0] for item in selected], np.asarray([item[1] for item in selected]))

    all_values = np.concatenate([values for _, values in grouped.values()])
    zoom_low = max(0.0, float(all_values.min()) - 0.00015)
    zoom_high = min(1.0, float(all_values.max()) + 0.00005)
    width, height = 2200, 1750
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 32), args.title, fill="#172033", font=font(38, True))
    draw.text(
        (90, 84),
        f"38#共{len(rows)}条｜2023-01-01 至 2025-09-15｜冻结阈值 {args.threshold:.4f}｜全部样本高于阈值",
        fill="#64748b", font=font(23),
    )

    overview_bounds = (70, 135, 2130, 650)
    overview_left, overview_top, overview_right, overview_bottom = overview_bounds
    draw.rectangle(overview_bounds, fill="#f8fafc", outline="#cbd5e1", width=2)
    x0, x1, y0, y1 = overview_left + 95, overview_right - 28, overview_top + 58, overview_bottom - 68
    all_times = [time for times, _ in grouped.values() for time in times]
    start, end = min(all_times), max(all_times)
    seconds = (end - start).total_seconds()
    def ox(time: datetime) -> float:
        return x0 + (time - start).total_seconds() / seconds * (x1 - x0)
    def oy(value: float) -> float:
        return y1 - value * (y1 - y0)
    draw.text((overview_left + 18, overview_top + 12), "全概率范围", fill="#172033", font=font(23, True))
    for index in range(5):
        value = index / 4
        y = oy(value)
        draw.line((x0, y, x1, y), fill="#d8e0e8", width=1)
        draw.text((x0 - 62, y - 10), f"{value:.2f}", fill="#334155", font=font(17))
    for index in range(5):
        fraction = index / 4
        time = start + (end - start) * fraction
        x = x0 + fraction * (x1 - x0)
        draw.line((x, y0, x, y1), fill="#d8e0e8", width=1)
        draw.text((x - 35, y1 + 15), time.strftime("%Y-%m"), fill="#334155", font=font(17))
    threshold_y = oy(args.threshold)
    for x in range(int(x0), int(x1), 28):
        draw.line((x, threshold_y, min(x + 14, x1), threshold_y), fill="#dc2626", width=3)
    draw.text((x1 - 210, threshold_y - 28), f"阈值 {args.threshold:.4f}", fill="#b91c1c", font=font(18, True))
    for sensor in sensors:
        times, values = grouped[sensor]
        draw.line([(ox(t), oy(float(v))) for t, v in zip(times, values)], fill=colors[sensor], width=2)
    legend_x = x0
    for sensor in sensors:
        draw.line((legend_x, y0 + 10, legend_x + 28, y0 + 10), fill=colors[sensor], width=4)
        draw.text((legend_x + 36, y0 - 3), f"{sensor}（{len(grouped[sensor][0])}条）", fill="#334155", font=font(17))
        legend_x += 270

    panel_bounds = [(70, 700, 1080, 1185), (1120, 700, 2130, 1185),
                    (70, 1225, 1080, 1710), (1120, 1225, 2130, 1710)]
    for sensor, bounds in zip(sensors, panel_bounds):
        times, values = grouped[sensor]
        title = f"{sensor}｜中位数 {np.median(values):.6f}｜样本 {len(values)}"
        draw_panel(draw, bounds, times, values, colors[sensor], title, zoom_low, zoom_high, True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


if __name__ == "__main__":
    main()
