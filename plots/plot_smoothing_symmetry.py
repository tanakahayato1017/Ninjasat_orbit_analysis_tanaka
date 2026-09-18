"""平滑化対称化感度: 非対称/対称/無平滑のラグ曲線比較 + ピークラグ差サマリ。

対応TODO: #20 (EUV/proxy間の非対称平滑化がクロス相関に与える影響, R1)

入力:
  data/figure_data/smoothing_symmetry.csv        (14_smoothing_symmetry.py, ラグ曲線)
  data/figure_data/smoothing_symmetry_peaks.csv  (同, ピーク表+asymmetric基準からの
    ピークラグの変化量 delta_peak_vs_asymmetric_h)

出力: figures/smoothing_symmetry.png / .pdf
  上段: 代表2波長 (304nm=EUV代表, 1216nm=Ly-alpha/FUV代表, geomag=all) の
        3条件 (asymmetric=現行10相当 / symmetric / none) ラグ曲線比較。
  下段: 7波長 x 2 geomag条件について、symmetric/noneのピークラグが
        asymmetric基準からどれだけ動いたか (delta_peak_vs_asymmetric_h) の
        サマリ。+-6h の判定閾値を縦線で表示。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_smoothing_symmetry.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CORR = ROOT / "data" / "figure_data" / "smoothing_symmetry.csv"
IN_PEAKS = ROOT / "data" / "figure_data" / "smoothing_symmetry_peaks.csv"
OUT = ROOT / "figures" / "smoothing_symmetry"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
WL_LABEL = {"256": "25.6", "284": "28.4", "304": "30.4",
            "1175": "117.5", "1216": "121.6", "1335": "133.5", "1405": "140.5"}
REP_WAVELENGTHS = ["304", "1216"]
REP_TITLES = {"304": "30.4 nm (EUV, He II)", "1216": "121.6 nm (Ly-alpha, FUV)"}

CONDITIONS = ["asymmetric", "symmetric", "none"]
COND_STYLE = {
    "asymmetric": dict(color="#1f77b4", lw=1.6, ls="-",
                        label="asymmetric (current: proxy=SG13, EUV=raw)"),
    "symmetric": dict(color="#2ca02c", lw=1.6, ls="--",
                       label="symmetric (proxy=SG13, EUV=SG13)"),
    "none": dict(color="#999999", lw=1.3, ls=":",
                 label="none (both raw)"),
}
DELTA_THRESHOLD_H = 6.0
PEAK_DOT_STYLE = {"symmetric": ("D", "#2ca02c"), "none": ("s", "#999999")}

corr = pd.read_csv(IN_CORR)
corr["wavelength"] = corr["wavelength"].astype(str)
peaks = pd.read_csv(IN_PEAKS)
peaks["wavelength"] = peaks["wavelength"].astype(str)

fig = plt.figure(figsize=(13, 9.5), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.12, wspace=0.22)

# ---------------- 上段: 代表2波長のラグ曲線 (geomag=all) ----------------
for i, wl in enumerate(REP_WAVELENGTHS):
    ax = fig.add_subplot(gs[0, i])
    ax.axhline(0, color="gray", lw=0.6, zorder=0)
    for cond in CONDITIONS:
        s = corr[(corr["condition"] == cond) & (corr["geomag"] == "all") &
                 (corr["wavelength"] == wl)].sort_values("lag_hours")
        style = COND_STYLE[cond]
        ax.plot(s["lag_hours"], s["spearman_r"], **style, alpha=0.9)
        prow = peaks[(peaks["condition"] == cond) & (peaks["geomag"] == "all") &
                     (peaks["wavelength"] == wl)]
        if len(prow):
            ax.plot(prow["peak_lag_hours"], prow["peak_r"], "o", ms=7,
                     mfc=style["color"], mec="black", mew=0.6, zorder=5)
    ax.set_xlabel("lag $L$ [h]")
    ax.set_ylabel("Spearman $r$")
    ax.set_title(f"{REP_TITLES[wl]}, geomag=all", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 120)
    if i == 0:
        ax.legend(fontsize=7.5, loc="lower right")

# ---------------- 下段: ピークラグ差のサマリ ----------------
ax = fig.add_subplot(gs[1, :])
geomags = ["all", "quiet_kp4"]
n_wl = len(WAVELENGTHS)
y_positions = {}
ypos = 0
group_gap = 1.0
yticks, yticklabels = [], []
for geomag in geomags:
    for wl in WAVELENGTHS:
        y_positions[(geomag, wl)] = ypos
        yticks.append(ypos)
        yticklabels.append(f"{WL_LABEL[wl]} nm")
        ypos += 1
    ypos += group_gap

ax.axvline(0, color="gray", lw=0.8, zorder=0)
ax.axvspan(-DELTA_THRESHOLD_H, DELTA_THRESHOLD_H, color="0.85", alpha=0.5, zorder=0)
ax.axvline(-DELTA_THRESHOLD_H, color="0.5", lw=0.7, ls="--", zorder=0)
ax.axvline(DELTA_THRESHOLD_H, color="0.5", lw=0.7, ls="--", zorder=0)

for cond in ["symmetric", "none"]:
    marker, color = PEAK_DOT_STYLE[cond]
    for geomag in geomags:
        sub = peaks[(peaks["condition"] == cond) & (peaks["geomag"] == geomag)]
        for wl in WAVELENGTHS:
            row = sub[sub["wavelength"] == wl]
            if not len(row):
                continue
            d = row["delta_peak_vs_asymmetric_h"].iloc[0]
            if pd.isna(d):
                continue
            y = y_positions[(geomag, wl)]
            offset = -0.15 if cond == "symmetric" else 0.15
            ax.plot(d, y + offset, marker, ms=7, mfc=color, mec="black",
                    mew=0.5, alpha=0.85,
                    label=(f"{cond} vs asymmetric" if (geomag == "all" and wl == WAVELENGTHS[0])
                           else None))

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=8)
ax.invert_yaxis()
XLIM = (-35, 15)
# geomag グループ境界に区切りラベル (データ座標を使い、constrained_layoutで
# 軸外のaxes-fraction文字が切り詰められるのを避ける)
ax.text(XLIM[0] + 1, y_positions[("all", WAVELENGTHS[0])] - 0.9, "geomag=all",
        fontsize=8.5, fontweight="bold", ha="left")
ax.text(XLIM[0] + 1, y_positions[("quiet_kp4", WAVELENGTHS[0])] - 0.9,
        "geomag=quiet_kp4", fontsize=8.5, fontweight="bold", ha="left")
ax.set_xlabel(r"$\Delta$ peak lag vs asymmetric (current method) [h]")
ax.set_title(
    "Peak-lag shift when EUV smoothing is made symmetric (diamond) or removed "
    "entirely (square), relative to the current asymmetric method\n"
    "gray band = |shift| < 6h (not judged meaningful); symmetric shifts stay "
    "mostly inside the band, 'none' shifts are large and one-sided", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left" if False else "lower left")
ax.grid(alpha=0.25, axis="x")
ax.set_xlim(-35, 15)

fig.suptitle("Smoothing symmetry sensitivity: asymmetric (current) vs symmetric vs no smoothing",
             fontsize=12.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
