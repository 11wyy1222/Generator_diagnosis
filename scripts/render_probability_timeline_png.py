from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render probability over time as a PNG")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--title", default="时间—异常概率曲线")
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.parquet)
    columns = ["acquisition_time", "abnormal_probability"]
    has_target = "target" in parquet.schema_arrow.names
    if has_target:
        columns.append("target")
    rows = parquet.read(columns=columns).to_pylist()
    points = sorted(
        (
            datetime.fromisoformat(str(row["acquisition_time"])),
            float(row["abnormal_probability"]),
            int(row["target"]) if has_target else None,
        )
        for row in rows
    )
    if not points:
        raise ValueError("the parquet file contains no probability points")

    width, height = 1800, 900
    left, right, top, bottom = 120, 55, 115, 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    background = "#f8fafc"
    grid = "#d9e1ea"
    axis = "#334155"
    text = "#172033"
    muted = "#64748b"
    line_color = "#1667b1"
    hit_color = "#1667b1"
    miss_color = "#d14343"
    threshold_color = "#d97706"

    draw.rectangle((left, top, width - right, height - bottom), fill=background)
    start, end = points[0][0], points[-1][0]
    total_seconds = max((end - start).total_seconds(), 1.0)

    def x_pos(time: datetime) -> float:
        return left + (time - start).total_seconds() / total_seconds * plot_width

    def y_pos(probability: float) -> float:
        return top + (1.0 - probability) * plot_height

    for index in range(6):
        probability = index / 5
        y = y_pos(probability)
        draw.line((left, y, width - right, y), fill=grid, width=2)
        label = f"{probability:.1f}"
        box = draw.textbbox((0, 0), label, font=font(25))
        draw.text((left - 18 - (box[2] - box[0]), y - 14), label, fill=axis, font=font(25))

    tick_count = 7
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        time = start + (end - start) * fraction
        x = left + plot_width * fraction
        draw.line((x, top, x, height - bottom), fill=grid, width=1)
        label = time.strftime("%Y-%m-%d")
        box = draw.textbbox((0, 0), label, font=font(22))
        label_width = box[2] - box[0]
        label_x = min(max(x - label_width / 2, 8), width - label_width - 8)
        draw.text((label_x, height - bottom + 18), label, fill=axis, font=font(22))

    draw.line((left, top, left, height - bottom), fill=axis, width=3)
    draw.line((left, height - bottom, width - right, height - bottom), fill=axis, width=3)

    threshold_y = y_pos(args.threshold)
    dash = 16
    cursor = left
    while cursor < width - right:
        draw.line((cursor, threshold_y, min(cursor + dash, width - right), threshold_y), fill=threshold_color, width=4)
        cursor += dash * 2
    threshold_label = f"冻结阈值 {args.threshold:.4f}"
    threshold_box = draw.textbbox((0, 0), threshold_label, font=font(24, bold=True))
    label_x = width - right - (threshold_box[2] - threshold_box[0]) - 14
    label_y = threshold_y - (threshold_box[3] - threshold_box[1]) - 12
    draw.rounded_rectangle(
        (label_x - 8, label_y - 5, width - right, threshold_y - 4),
        radius=6,
        fill="white",
        outline=threshold_color,
        width=2,
    )
    draw.text((label_x, label_y), threshold_label, fill=threshold_color, font=font(24, bold=True))

    xy = [(x_pos(time), y_pos(probability)) for time, probability, _ in points]
    draw.line(xy, fill=line_color, width=3, joint="curve")
    for (x, y), (_, probability, target) in zip(xy, points):
        prediction = probability >= args.threshold
        correct = prediction == bool(target) if target is not None else prediction
        color = hit_color if correct else miss_color
        radius = 3 if correct else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    if has_target:
        tp = sum(probability >= args.threshold and target == 1 for _, probability, target in points)
        fn = sum(probability < args.threshold and target == 1 for _, probability, target in points)
        fp = sum(probability >= args.threshold and target == 0 for _, probability, target in points)
        tn = sum(probability < args.threshold and target == 0 for _, probability, target in points)
        accuracy = (tp + tn) / len(points)
        subtitle = (
            f"每个点代表一条波形｜样本数 {len(points)}｜异常检出 {tp}/{tp + fn}｜"
            f"正常误报 {fp}/{fp + tn}｜准确率 {accuracy:.2%}"
        )
        good_label, bad_label = "分类正确", "分类错误"
    else:
        hits = sum(probability >= args.threshold for _, probability, _ in points)
        subtitle = f"每个点代表一条波形｜样本数 {len(points)}｜阈值命中 {hits}/{len(points)}（{hits / len(points):.2%}）"
        good_label, bad_label = "达到阈值", "未达到阈值"
    draw.text((left, 28), args.title, fill=text, font=font(36, bold=True))
    draw.text((left, 75), subtitle, fill=muted, font=font(24))
    draw.text((left + plot_width / 2 - 55, height - 42), "采集时间", fill=axis, font=font(26, bold=True))
    draw.text((25, top + plot_height / 2 - 18), "异常概率", fill=axis, font=font(26, bold=True))

    legend_y = 74
    legend_x = width - 570
    draw.ellipse((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=hit_color)
    draw.text((legend_x + 20, legend_y - 10), good_label, fill=axis, font=font(22))
    legend_x += 150
    draw.ellipse((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=miss_color)
    draw.text((legend_x + 20, legend_y - 10), bad_label, fill=axis, font=font(22))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
