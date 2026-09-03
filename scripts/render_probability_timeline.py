from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a probability-over-time HTML fragment")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    args = parser.parse_args()

    columns = ["sample_id", "acquisition_time", "rpm", "abnormal_probability"]
    rows = pq.read_table(args.parquet, columns=columns).to_pylist()
    rows.sort(key=lambda row: str(row["acquisition_time"]))
    points = [
        {
            "t": str(row["acquisition_time"]).replace(" ", "T"),
            "p": round(float(row["abnormal_probability"]), 7),
            "rpm": round(float(row["rpm"]), 1),
            "id": str(row["sample_id"]),
        }
        for row in rows
    ]

    template = """<div id="f14-probability-timeline-root">
  <div id="f14-probability-timeline-chart" role="img" aria-label="F14每条波形按采集时间排列的异常概率曲线，并标出冻结评估阈值"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(() => {
  const root = document.getElementById('f14-probability-timeline-root');
  const chart = document.getElementById('f14-probability-timeline-chart');
  const points = __POINTS__;
  const threshold = __THRESHOLD__;
  const styles = getComputedStyle(root);
  const foreground = styles.getPropertyValue('--foreground').trim();
  const muted = styles.getPropertyValue('--muted-foreground').trim();
  const border = styles.getPropertyValue('--border').trim();
  const series = styles.getPropertyValue('--viz-series-1').trim();
  const thresholdColor = styles.getPropertyValue('--viz-series-2').trim();
  const x = points.map((point) => point.t);
  const y = points.map((point) => point.p);
  const custom = points.map((point) => [point.id, point.rpm]);
  Plotly.newPlot(chart, [{
    x,
    y,
    customdata: custom,
    type: 'scattergl',
    mode: 'lines+markers',
    name: '异常概率',
    line: { color: series, width: 1.5 },
    marker: { color: series, size: 4 },
    hovertemplate: '时间 %{x}<br>异常概率 %{y:.4f}<br>RPM %{customdata[1]:.1f}<br>样本 %{customdata[0]}<extra></extra>'
  }], {
    autosize: true,
    height: 440,
    margin: { l: 58, r: 24, t: 18, b: 58 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: foreground },
    hovermode: 'closest',
    showlegend: false,
    xaxis: {
      title: '采集时间',
      color: foreground,
      gridcolor: border,
      linecolor: border,
      rangeslider: { visible: true, thickness: 0.09 },
      type: 'date'
    },
    yaxis: {
      title: '异常概率',
      color: foreground,
      gridcolor: border,
      linecolor: border,
      range: [0, 1.02],
      tickformat: '.1f'
    },
    shapes: [{
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      yref: 'y',
      y0: threshold,
      y1: threshold,
      line: { color: thresholdColor, width: 2, dash: 'dash' }
    }],
    annotations: [{
      xref: 'paper',
      x: 1,
      xanchor: 'right',
      yref: 'y',
      y: threshold,
      yanchor: 'bottom',
      text: `冻结阈值 ${threshold.toFixed(4)}`,
      showarrow: false,
      font: { color: muted }
    }]
  }, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d']
  });
  new ResizeObserver(() => Plotly.Plots.resize(chart)).observe(root);
})();
</script>
"""
    fragment = template.replace("__POINTS__", json.dumps(points, ensure_ascii=False))
    fragment = fragment.replace("__THRESHOLD__", repr(float(args.threshold)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")


if __name__ == "__main__":
    main()
