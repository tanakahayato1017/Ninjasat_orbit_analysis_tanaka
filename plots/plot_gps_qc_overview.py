"""GPSテレメトリQC後データの品質全体像 (図1: gps_qc_overview)。

対応TODO: #1 の前段可視化

入力:
  data/figure_data/gps_qc_overview_daily.csv       (日毎サンプル数)
  data/figure_data/gps_qc_overview_gaps.csv         (6h超の欠測ギャップ一覧)
  data/figure_data/gps_qc_overview_sats_hist.csv    (追尾衛星数ヒストグラム)
  data/figure_data/gps_qc_overview_hdop_hist.csv    (HDOPヒストグラム)
出力: figures/gps_qc_overview.png / .pdf

レイアウト:
  (a) 全期間の日毎QC通過サンプル数 (棒グラフ)。6h超の欠測ギャップのうち
      最大2件 (2024-05-20〜29の9日間, 2024-09-09〜12の約48h) を薄い縦帯+注記で強調。
  (b) 追尾衛星数のヒストグラム
  (c) HDOPのヒストグラム

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行: .venv/Scripts/python plots/plot_gps_qc_overview.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN_DAILY = ROOT / "data" / "figure_data" / "gps_qc_overview_daily.csv"
IN_GAPS = ROOT / "data" / "figure_data" / "gps_qc_overview_gaps.csv"
IN_SATS = ROOT / "data" / "figure_data" / "gps_qc_overview_sats_hist.csv"
IN_HDOP = ROOT / "data" / "figure_data" / "gps_qc_overview_hdop_hist.csv"
OUT = ROOT / "figures" / "gps_qc_overview"

N_GAPS_ANNOTATE = 2  # 最大の欠測ギャップのうち注記するもの (9日ギャップ, 9月の~48hギャップ)

daily = pd.read_csv(IN_DAILY, parse_dates=["date"])
gaps = pd.read_csv(IN_GAPS, parse_dates=["gap_start", "gap_end"])
sats_hist = pd.read_csv(IN_SATS)
hdop_hist = pd.read_csv(IN_HDOP)

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.25)

# --- (a) 日毎QC通過サンプル数 ---
ax = fig.add_subplot(gs[0, :])
ax.bar(daily["date"], daily["n_samples"], width=0.9, color="#4C72B0", align="edge")
ax.set_ylabel("QC-passed samples / day")
ax.set_title("(a) NinjaSat GPS telemetry: daily QC-passed sample count, "
              "2024-03-31 to 2024-11-30")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylim(0, daily["n_samples"].max() * 1.15)

top_gaps = gaps.nlargest(N_GAPS_ANNOTATE, "duration_hours").sort_values("gap_start")
for _, g in top_gaps.iterrows():
    ax.axvspan(g["gap_start"], g["gap_end"], color="crimson", alpha=0.18, lw=0)
    mid = g["gap_start"] + (g["gap_end"] - g["gap_start"]) / 2
    days = g["duration_hours"] / 24.0
    label = (f"{days:.0f}d gap\n{g['gap_start'].date()}"
             if g["duration_hours"] >= 72
             else f"~{g['duration_hours']:.0f}h gap\n{g['gap_start'].date()}")
    ax.annotate(label, xy=(mid, daily["n_samples"].max() * 1.0),
                ha="center", va="bottom", fontsize=8, color="crimson")
ax.grid(alpha=0.3, axis="y")
n_other_gaps = len(gaps) - len(top_gaps)
fig.text(0.99, 0.545,
         f"{len(gaps)} gaps > 6h total; {n_other_gaps} smaller ones "
         "(daily/weekend downlink pattern, up to ~26h) not individually marked",
         ha="right", va="top", fontsize=7, color="gray")

# --- (b) 追尾衛星数ヒストグラム ---
ax = fig.add_subplot(gs[1, 0])
centers = (sats_hist["bin_left"] + sats_hist["bin_right"]) / 2
ax.bar(centers, sats_hist["count"], width=(sats_hist["bin_right"] - sats_hist["bin_left"]),
       color="#55A868", edgecolor="white", linewidth=0.3)
ax.set_xlabel("gpsNumOfSatsTracked")
ax.set_ylabel("count")
ax.set_title("(b) Tracked satellites")
ax.grid(alpha=0.3, axis="y")

# --- (c) HDOPヒストグラム ---
ax = fig.add_subplot(gs[1, 1])
centers = (hdop_hist["bin_left"] + hdop_hist["bin_right"]) / 2
ax.bar(centers, hdop_hist["count"], width=(hdop_hist["bin_right"] - hdop_hist["bin_left"]),
       color="#DD8452", edgecolor="white", linewidth=0.3)
ax.set_xlabel("gpsHDilutionOfPos")
ax.set_ylabel("count")
ax.set_title("(c) Horizontal dilution of precision")
ax.set_xlim(0, hdop_hist.loc[hdop_hist["count"] > 0, "bin_right"].max())
ax.grid(alpha=0.3, axis="y")

fig.suptitle("GPS telemetry QC overview (input to specific-energy pipeline)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
fig.savefig(OUT.with_suffix(".pdf"))
print(f"wrote {OUT.with_suffix('.png')}")
print(f"wrote {OUT.with_suffix('.pdf')}")
