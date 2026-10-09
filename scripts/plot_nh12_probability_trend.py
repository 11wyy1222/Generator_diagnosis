from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

DEFAULT_DIR = Path("runs/dfig_diagnosis_v2_20260915_seed2026/test_nh12")


def font(size: int, bold: bool = False):
    names = ["msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_DIR / "predictions_test.parquet")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_DIR / "metrics_test.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "nh12_abnormal_probability_trend.png")
    args = parser.parse_args()

    data = pd.read_parquet(args.predictions)
    required = {"sample_id", "object_id", "acquisition_time", "target", "range_position", "abnormal_probability"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if set(data.object_id.unique()) != {"DFIG_NEW_NH_12"}:
        raise ValueError("input must contain only DFIG_NEW_NH_12")
    data.acquisition_time = pd.to_datetime(data.acquisition_time, errors="raise")
    data = data.sort_values(["acquisition_time", "sample_id"])
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["overall"]["threshold"])

    scale, w, h = 2, 1600, 800
    image = Image.new("RGB", (w * scale, h * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, right, top, bottom = 110 * scale, 1545 * scale, 125 * scale, 665 * scale
    draw.rectangle((left, top, right, bottom), fill="#fbfcfe")
    t0, t1 = data.acquisition_time.min(), data.acquisition_time.max()
    seconds = (t1 - t0).total_seconds()
    xof = lambda t: left + (t - t0).total_seconds() / seconds * (right - left)
    yof = lambda p: bottom - max(0.0, min(1.0, float(p))) * (bottom - top)

    small, tick, axis, title = font(15 * scale), font(16 * scale), font(18 * scale), font(30 * scale, True)
    regions = [
        ("正常范围", data.target.eq(0), (219, 234, 254, 108)),
        ("异常早期", data.range_position.eq("early"), (255, 237, 213, 108)),
        ("异常中期", data.range_position.eq("middle"), (254, 215, 170, 108)),
        ("异常晚期", data.range_position.eq("late"), (253, 186, 116, 108)),
    ]
    for label, mask, color in regions:
        part = data.loc[mask]
        if part.empty:
            continue
        x1, x2 = xof(part.acquisition_time.min()), xof(part.acquisition_time.max())
        draw.rectangle((x1, top, x2, bottom), fill=color)
        text = f"{label}  n={len(part)}"
        box = draw.textbbox((0, 0), text, font=small)
        draw.text(((x1 + x2 - box[2]) / 2, top - 30 * scale), text, font=small, fill="#374151")

    for n in range(0, 11, 2):
        value, y = n / 10, yof(n / 10)
        draw.line((left, y, right, y), fill="#d1d5db", width=scale)
        text = f"{value:.1f}"
        box = draw.textbbox((0, 0), text, font=tick)
        draw.text((left - box[2] - 18 * scale, y - (box[3] - box[1]) / 2), text, font=tick, fill="#4b5563")

    months = pd.date_range(t0.normalize().replace(day=1), t1.normalize(), freq="MS")
    for stamp in months[months >= t0]:
        x = xof(stamp)
        draw.line((x, bottom, x, bottom + 7 * scale), fill="#6b7280", width=scale)
        text = stamp.strftime("%Y-%m")
        box = draw.textbbox((0, 0), text, font=tick)
        draw.text((x - box[2] / 2, bottom + 14 * scale), text, font=tick, fill="#4b5563")

    colors = {0: "#2563eb", 1: "#d97706"}
    for target_value in (0, 1):
        part = data.loc[data.target.eq(target_value)]
        points = [(xof(row.acquisition_time), yof(row.abnormal_probability)) for row in part.itertuples()]
        if len(points) > 1:
            draw.line(points, fill=colors[target_value], width=scale)
        for x, y in points:
            r = 2.2 * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[target_value])

    daily = data.set_index("acquisition_time").abnormal_probability.resample("1D").median().dropna()
    rolling = daily.rolling("7D", min_periods=1, center=True).median()
    smooth = [(xof(stamp), yof(value)) for stamp, value in rolling.items()]
    if len(smooth) > 1:
        draw.line(smooth, fill="#111827", width=3 * scale, joint="curve")

    y = yof(threshold)
    x = left
    while x < right:
        draw.line((x, y, min(x + 12 * scale, right), y), fill="#7c3aed", width=2 * scale)
        x += 20 * scale
    draw.text((left + 10 * scale, y - 28 * scale), f"冻结评估阈值 {threshold:.3f}", font=small, fill="#6d28d9")
    draw.line((left, top, left, bottom), fill="#6b7280", width=2 * scale)
    draw.line((left, bottom, right, bottom), fill="#6b7280", width=2 * scale)

    heading = "双馈内黄12#异常概率时间趋势（V2，seed 2026）"
    box = draw.textbbox((0, 0), heading, font=title)
    draw.text(((w * scale - box[2]) / 2, 26 * scale), heading, font=title, fill="#111827")
    draw.text((14 * scale, (top + bottom) / 2), "异常概率", font=axis, fill="#111827")
    draw.text(((left + right) / 2 - 36 * scale, bottom + 54 * scale), "采集时间", font=axis, fill="#111827")

    stages = metrics["by_range_position"]
    summary = "   ".join(
        f"{label}召回 {stages[key]['recall_range_label'] * 100:.1f}%"
        for label, key in (("早期", "early"), ("中期", "middle"), ("晚期", "late"))
    )
    draw.rounded_rectangle((left + 14 * scale, top + 12 * scale, left + 650 * scale, top + 55 * scale), 8 * scale, fill="white", outline="#d1d5db", width=scale)
    draw.text((left + 25 * scale, top + 19 * scale), summary, font=small, fill="#111827")

    legend = [("#2563eb", "正常范围逐波形"), ("#d97706", "异常范围逐波形"), ("#111827", "7日滚动中位数"), ("#7c3aed", "冻结阈值")]
    lx, ly = left, bottom + 92 * scale
    for color, text in legend:
        draw.line((lx, ly + 9 * scale, lx + 22 * scale, ly + 9 * scale), fill=color, width=3 * scale)
        draw.text((lx + 29 * scale, ly), text, font=small, fill="#374151")
        lx += (len(text) * 17 + 80) * scale
    note = "注：异常标签来自业务确认的时间范围，不代表范围内每条波形均具有可见故障特征；曲线不应直接解释为故障严重度。"
    draw.text((left, h * scale - 34 * scale), note, font=small, fill="#4b5563")

    image = image.resize((w, h), Image.Resampling.LANCZOS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Source", str(args.predictions))
    image.save(args.output, pnginfo=metadata, optimize=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
