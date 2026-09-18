"""Dstベースの静穏期マスクのクロスチェック図 (2段構成)。

対応TODO: #8 (地磁気活動の定量的評価 — R2 Minor#1のKp/Ap/Dst列挙のうちDst部分)

入力:
  data/processed/dst_hourly.csv          (18_dst_quiet_check.py, 1h Dst)
  data/figure_data/dst_quiet_mask_3h.csv (18_dst_quiet_check.py, 3hビンquiet_dst30/50)
  data/figure_data/dst_storm_events_kp6.csv (18_dst_quiet_check.py, Kp>=6イベント)
  data/figure_data/dst_kp_peak_comparison.csv (18_dst_quiet_check.py,
    window=13/period=fullでのピークラグ: all/quiet_kp4/quiet_kp3/quiet_dst50/quiet_dst30)

出力: figures/dst_quiet_check.png / .pdf (2段構成)
  (a) 全期間 (2024-04-01〜2024-11-30) のDst 1h時系列。quiet_dst50 (直近48h最小
      Dst > -50 nT) が成立する3h区間を緑帯で重ね、Kp>=6の磁気嵐イベントの
      ピーク日付を注記する。
  (b) 波長別ピークラグ [h] (window=13, period=full) を条件
      (all / quiet_kp4 / quiet_kp3 / quiet_dst50 / quiet_dst30) を横軸に、
      波長ごとに折れ線でつないだもの。灰帯は旧論文のEUV帯(30-36h)・FUV帯(48-54h)。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_dst_check.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DST_HOURLY = ROOT / "data" / "processed" / "dst_hourly.csv"
IN_MASK_3H = ROOT / "data" / "figure_data" / "dst_quiet_mask_3h.csv"
IN_STORM_EVENTS = ROOT / "data" / "figure_data" / "dst_storm_events_kp6.csv"
IN_COMPARISON = ROOT / "data" / "figure_data" / "dst_kp_peak_comparison.csv"
OUT = ROOT / "figures" / "dst_quiet_check"

PERIOD_START = pd.Timestamp("2024-04-01")
PERIOD_END = pd.Timestamp("2024-11-30 21:00:00")  # last 3h bin start of the period
BIN_TD = pd.Timedelta("3h")
STORM_LABEL_MIN_PEAK_KP = 7.0  # 混雑を避けるため、注記はこれ以上のイベントのみ

WAVELENGTHS = [256, 284, 304, 1175, 1216, 1335, 1405]
WAVELENGTH_LABELS = {256: "25.6 nm", 284: "28.4 nm", 304: "30.4 nm",
                      1175: "117.5 nm", 1216: "121.6 nm",
                      1335: "133.5 nm", 1405: "140.5 nm"}
CATEGORIES = ["all", "quiet_kp4", "quiet_kp3", "quiet_dst50", "quiet_dst30"]
EUV_BAND = (30, 36)   # candidate short-wavelength response window [h]
FUV_BAND = (48, 54)   # candidate long-wavelength response window [h]

dst_hourly = pd.read_csv(IN_DST_HOURLY, parse_dates=["datetime"])
mask_3h = pd.read_csv(IN_MASK_3H, parse_dates=["datetime"])
storms = pd.read_csv(IN_STORM_EVENTS, parse_dates=["start", "end", "peak_time"])
comparison = pd.read_csv(IN_COMPARISON)

dst_hourly = dst_hourly[
    (dst_hourly["datetime"] >= PERIOD_START) &
    (dst_hourly["datetime"] <= PERIOD_END + BIN_TD)
].copy()


def quiet_intervals(d: pd.DataFrame, col: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """連続するcol=Trueの3hビンをまとめて[start,end)区間にする
    (plot_geomag_overview.py と同じアルゴリズムをここで独立に再実装)。"""
    times = d.loc[d[col], "datetime"].sort_values().to_numpy()
    if len(times) == 0:
        return []
    out = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > np.timedelta64(3, "h"):
            out.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
            start = t
        prev = t
    out.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
    return out


qintervals = quiet_intervals(mask_3h, "quiet_dst50")
print(f"quiet_dst50 intervals: {len(qintervals)}, "
      f"quiet fraction: {mask_3h['quiet_dst50'].mean():.1%}")
print(f"storm (Kp>=6) events in period: {len(storms)}")

fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                          gridspec_kw={"height_ratios": [1.2, 1]})

# ---------------- (a) 全期間Dst時系列 + quiet_dst50マスク + 磁気嵐注記 ----------------
ax = axes[0]
for s, e in qintervals:
    ax.axvspan(s, e, color="#2ca02c", alpha=0.12, lw=0)
ax.plot(dst_hourly["datetime"], dst_hourly["Dst_nT"], color="#4C72B0",
        lw=0.7, alpha=0.9)
ax.axhline(-50, color="0.4", lw=0.8, ls=":", alpha=0.8,
           label="Dst=-50 nT (quiet_dst50 threshold)")
ax.axhline(-30, color="0.55", lw=0.7, ls=":", alpha=0.6,
           label="Dst=-30 nT (quiet_dst30 threshold)")
ax.axhline(0, color="0.7", lw=0.5)

label_toggle = 0
for _, e in storms.iterrows():
    if e["peak_kp"] < STORM_LABEL_MIN_PEAK_KP:
        continue
    idx = (dst_hourly["datetime"] - e["peak_time"]).abs().idxmin()
    y_at_peak = dst_hourly.loc[idx, "Dst_nT"]
    y_text = -430 if label_toggle % 2 == 0 else -480
    label_toggle += 1
    ax.annotate(f"{e['peak_time'].month}/{e['peak_time'].day}\n"
                f"(Kp={e['peak_kp']:.0f})",
                xy=(e["peak_time"], y_at_peak), xytext=(e["peak_time"], y_text),
                fontsize=7.5, ha="center", color="#7f0000",
                arrowprops=dict(arrowstyle="-", color="#7f0000", lw=0.6, alpha=0.7))

ax.set_ylim(-520, 60)
ax.set_ylabel("Dst [nT] (1h, OMNI2)")
ax.set_title(
    "(a) Full-period Dst index (2024-04-01 to 2024-11-30)\n"
    "green bands = quiet windows (trailing 48h min Dst > -50 nT, i.e. 'quiet_dst50'); "
    "dates = storm peak Kp>=" f"{STORM_LABEL_MIN_PEAK_KP:.0f} events (from geomag_3h.csv)",
    fontsize=9.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.25, axis="y")

# ---------------- (b) 波長別ピークラグの条件比較 ----------------
ax = axes[1]
ax.axhspan(*EUV_BAND, color="0.85", zorder=0,
           label=f"old-paper EUV band ({EUV_BAND[0]}-{EUV_BAND[1]}h)")
ax.axhspan(*FUV_BAND, color="0.75", zorder=0,
           label=f"old-paper FUV band ({FUV_BAND[0]}-{FUV_BAND[1]}h)")

x = np.arange(len(CATEGORIES))
cmap = plt.get_cmap("plasma")
colors = {wl: cmap(i / (len(WAVELENGTHS) - 1)) for i, wl in enumerate(WAVELENGTHS)}

comp = comparison.set_index("wavelength")
for wl in WAVELENGTHS:
    if wl not in comp.index:
        continue
    y = [comp.loc[wl, c] if c in comp.columns else np.nan for c in CATEGORIES]
    ax.plot(x, y, marker="o", ms=5, lw=1.4, color=colors[wl],
            label=WAVELENGTH_LABELS[wl])

ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES, rotation=15)
ax.set_ylabel("peak lag [h] (lag>=6h, window=13, period=full)")
ax.set_title(
    "(b) Peak lag by geomagnetic condition: Kp-based (all/quiet_kp4/quiet_kp3) "
    "vs. Dst-based (quiet_dst50/quiet_dst30)", fontsize=9.5)
ax.set_ylim(0, 60)
ax.grid(alpha=0.25, axis="y")
ax.legend(loc="upper center", fontsize=7.5, ncol=5, bbox_to_anchor=(0.5, -0.18))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
