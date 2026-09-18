"""Figure 6 (本文): ピークラグの平滑化窓幅・地磁気活動への感度、2パネル統合図。

対応TODO: #4 (Savitzky-Golayフィルタのアーティファクト検証), #8 (地磁気活動の定量的評価)
manuscripts.tex 452-455行目のFigure 6キャプションに対応する単一の2パネル図。

背景:
  従来は plot_euv_lag_scan.py が同じ元データから euv_peak_vs_window.png (窓幅依存) と
  euv_peak_quiet_vs_all.png (地磁気条件依存, 2パネル) という別々のファイルを出力していた。
  本文Figure 6はこの2つを (a)(b) の1枚の図にまとめたもの:
    (a) = euv_peak_vs_window.png と同内容 (全波長, full期間, geomag=all, 窓幅0-39点)。
    (b) = euv_peak_quiet_vs_all.png のうち window=13 (~39h, 旧解析の窓) のパネルのみを
          抜き出し、all / quiet(Kp48h,max<4) の2条件を比較 (quiet_kp3は本図には含めない)。

入力:
  data/figure_data/euv_lag_peaks.csv  (10_euv_lag_correlation.py の出力,
      wavelength x window x geomag x period のピークラグ表)

出力:
  figures/Figure6_combined.pdf / .png

すべて図データCSVから描く (再計算しない)。
実行:  .venv/Scripts/python plots/plot_figure6_combined.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PEAKS = ROOT / "data" / "figure_data" / "euv_lag_peaks.csv"
OUT_DIR = ROOT / "figures"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
EUV_BAND = ["256", "284", "304"]
FUV_BAND = ["1175", "1216", "1335", "1405"]
EUV_COLORS = dict(zip(EUV_BAND, ["#7f0000", "#d62728", "#ff9896"]))
FUV_COLORS = dict(zip(FUV_BAND, ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]))
COLORS = {**EUV_COLORS, **FUV_COLORS}
# 凡例はnm表記 (データ列キーはÅ值; "256 nm"等の10x誤表記を避ける)
WL_LABEL = {"256": "25.6 nm", "284": "28.4 nm", "304": "30.4 nm",
            "1175": "117.5 nm", "1216": "121.6 nm (Ly-a)",
            "1335": "133.5 nm", "1405": "140.5 nm"}

FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 13
FS_TITLE = 15
FS_SUBTITLE = 14
FS_PANEL = 16

peaks = pd.read_csv(IN_PEAKS)
peaks["wavelength"] = peaks["wavelength"].astype(str)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))

# ============================ (a) ピークラグ vs SG窓幅 ============================
sub = peaks[(peaks["period"] == "full") & (peaks["geomag"] == "all")]
for wl in WAVELENGTHS:
    s = sub[sub["wavelength"] == wl].sort_values("window")
    x = s["window"].replace(-1, 0)
    ax_a.plot(x, s["peak_lag_hours"], "-o", ms=6, lw=1.5, color=COLORS[wl],
              alpha=0.85, label=WL_LABEL[wl])

ax_a.set_xlabel("Savitzky-Golay window [points]\n(0 = no smoothing, 13 = 39-h smoothing window)",
                 fontsize=FS_LABEL)
ax_a.set_ylabel("peak lag [h]  (max Spearman r, lag>=6h)", fontsize=FS_LABEL)
ax_a.set_title("Peak lag vs SG smoothing-window width\n(full period, geomag=all)",
                fontsize=FS_SUBTITLE)
ax_a.legend(fontsize=FS_LEGEND, loc="lower right", ncol=2)
ax_a.grid(alpha=0.3)
ax_a.set_xticks(sorted(x.unique()))
ax_a.tick_params(labelsize=FS_TICK)
ax_a.text(0.02, 0.97, "(a)", transform=ax_a.transAxes, fontsize=FS_PANEL,
          fontweight="bold", va="top", ha="left")

# ==================== (b) window=13 のみ: geomag条件別ピークラグ ====================
geomag_order = ["all", "quiet_kp4"]
geomag_x = {g: i for i, g in enumerate(geomag_order)}
geomag_labels = ["all", "quiet\n(Kp$_{48h,max}$<4)"]

sub_b = peaks[(peaks["period"] == "full") & (peaks["window"] == 13) &
              (peaks["geomag"].isin(geomag_order))]
for wl in WAVELENGTHS:
    s = sub_b[sub_b["wavelength"] == wl].set_index("geomag").reindex(geomag_order)
    x = [geomag_x[g] + (WAVELENGTHS.index(wl) - 3) * 0.04 for g in geomag_order]
    ax_b.plot(x, s["peak_lag_hours"], "o-", ms=8, lw=1.6, color=COLORS[wl],
              alpha=0.85, label=WL_LABEL[wl])

ax_b.set_xticks(list(geomag_x.values()))
ax_b.set_xticklabels(geomag_labels, fontsize=FS_TICK)
ax_b.set_xlim(-0.4, 1.4)
ax_b.set_ylabel("peak lag [h]", fontsize=FS_LABEL)
ax_b.set_title("Peak lag: all-data vs. geomagnetically quiet\n(SG window=13, ~39h, full period)",
               fontsize=FS_SUBTITLE)
ax_b.legend(fontsize=FS_LEGEND, loc="upper right", ncol=2)
ax_b.grid(alpha=0.3)
ax_b.tick_params(labelsize=FS_TICK)
ax_b.text(0.02, 0.97, "(b)", transform=ax_b.transAxes, fontsize=FS_PANEL,
          fontweight="bold", va="top", ha="left")

fig.suptitle("Sensitivity of the peak lag to smoothing-window width and geomagnetic activity",
             fontsize=FS_TITLE)
fig.tight_layout(rect=(0, 0, 1, 0.93))

OUT = OUT_DIR / "Figure6_combined"
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
