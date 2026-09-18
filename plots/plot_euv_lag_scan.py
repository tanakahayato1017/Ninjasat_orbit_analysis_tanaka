"""EUV/FUV波長別ラグ相関スキャンの可視化 (TODO #4 アーティファクト判定用の3図)。

対応TODO: #4 (Savitzky-Golayフィルタのアーティファクト検証),
          #8 (地磁気活動の定量的評価)

入力:
  data/figure_data/euv_lag_correlation.csv  (10_euv_lag_correlation.py の出力)
  data/figure_data/euv_lag_peaks.csv        (同, ピークラグ表)
  data/figure_data/euv_lag_bootstrap.csv    (同, ブロックブートストラップCI)

出力:
  figures/euv_lag_curves.pdf / .png
      波長ごとの相関 vs ラグ曲線 (full期間, geomag=all, window={なし,13}を重ね描き)。
  figures/euv_peak_vs_window.pdf / .png
      波長ごとのピークラグ vs SG窓幅 (full, geomag=all)。ピークが窓幅に追従して
      動くならアーティファクトを示唆する。
  figures/euv_peak_quiet_vs_all.pdf / .png
      geomag条件別のピークラグ比較 (full, window={なし,13}の2パネル)。

すべて図データCSVから描く (再計算しない)。
実行:  .venv/Scripts/python plots/plot_euv_lag_scan.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CORR = ROOT / "data" / "figure_data" / "euv_lag_correlation.csv"
IN_PEAKS = ROOT / "data" / "figure_data" / "euv_lag_peaks.csv"
IN_BOOT = ROOT / "data" / "figure_data" / "euv_lag_bootstrap.csv"
OUT_DIR = ROOT / "figures"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
EUV_BAND = ["256", "284", "304"]       # 短波長側(EUV)
FUV_BAND = ["1175", "1216", "1335", "1405"]  # 長波長側(FUV)

# EUV帯は暖色系、FUV帯は寒色系でグラデーション
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
FS_TEXT = 11
FS_PANEL = 16

corr = pd.read_csv(IN_CORR)
peaks = pd.read_csv(IN_PEAKS)
boot = pd.read_csv(IN_BOOT)

# CSVのwavelength列は整数で保存されているため、WAVELENGTHS(文字列)と
# 比較できるように統一する (文字列のままフィルタすると全て空になる)
for _df in (corr, peaks, boot):
    _df["wavelength"] = _df["wavelength"].astype(str)


# ============================== 図1: ラグ曲線 (Figure 4) ==============================

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

panel_labels = ["(a)", "(b)"]
for ax, window, title, lab in zip(
        axes, [-1, 13], ["no smoothing", "SG window=13 (~39h)"], panel_labels):
    sub = corr[(corr["period"] == "full") & (corr["geomag"] == "all") &
               (corr["window"] == window)]
    for wl in WAVELENGTHS:
        s = sub[sub["wavelength"] == wl].sort_values("lag_hours")
        ax.plot(s["lag_hours"], s["spearman_r"], "-o", ms=3, lw=1.4,
                color=COLORS[wl], alpha=0.85, label=WL_LABEL[wl])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("lag L [h]  (EUV(t-L) vs proxy(t))", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_SUBTITLE)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 120)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

axes[0].set_ylabel("Spearman r", fontsize=FS_LABEL)
axes[1].legend(fontsize=FS_LEGEND, loc="lower right", ncol=2)
fig.suptitle(r"Lag correlation: EUV/FUV($t-L$) vs $(-dE/dt)$ proxy($t$), "
             "full period, geomag=all", fontsize=FS_TITLE)
fig.tight_layout(rect=(0, 0, 1, 0.88))

OUT1 = OUT_DIR / "euv_lag_curves"
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(f"{OUT1}.pdf")
fig.savefig(f"{OUT1}.png", dpi=160)
print(f"wrote {OUT1}.pdf / .png")
plt.close(fig)

# ======================== 図2: ピークラグ vs SG窓幅 ========================

fig, ax = plt.subplots(figsize=(9, 6.5))
sub = peaks[(peaks["period"] == "full") & (peaks["geomag"] == "all")]
for wl in WAVELENGTHS:
    s = sub[sub["wavelength"] == wl].sort_values("window")
    x = s["window"].replace(-1, 0)  # 表示上 -1(平滑化なし)を0にマップ
    ax.plot(x, s["peak_lag_hours"], "-o", ms=6, lw=1.5, color=COLORS[wl],
            alpha=0.85, label=WL_LABEL[wl])

ax.set_xlabel("Savitzky-Golay window [points] (0 = no smoothing)", fontsize=FS_LABEL)
ax.set_ylabel("peak lag [h]  (max Spearman r, lag>=6h)", fontsize=FS_LABEL)
ax.set_title("Peak lag vs SG smoothing window (full period, geomag=all)\n"
             "peak tracking the window width would indicate a filter artifact "
             "(TODO #4)", fontsize=FS_TITLE)
ax.legend(fontsize=FS_LEGEND, loc="best", ncol=2)
ax.grid(alpha=0.3)
ax.set_xticks(sorted(x.unique()))
ax.tick_params(labelsize=FS_TICK)

OUT2 = OUT_DIR / "euv_peak_vs_window"
fig.tight_layout()
fig.savefig(f"{OUT2}.pdf")
fig.savefig(f"{OUT2}.png", dpi=160)
print(f"wrote {OUT2}.pdf / .png")
plt.close(fig)

# ==================== 図3: geomag条件別のピークラグ比較 ====================

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
geomag_order = ["all", "quiet_kp4", "quiet_kp3"]
geomag_x = {g: i for i, g in enumerate(geomag_order)}
geomag_labels = ["all", "quiet\n(Kp$_{48h,max}$<4)", "quiet\n(Kp$_{48h,max}$<3)"]

for ax, window, title in zip(axes, [-1, 13], ["no smoothing", "SG window=13 (~39h)"]):
    sub = peaks[(peaks["period"] == "full") & (peaks["window"] == window)]
    for wl in WAVELENGTHS:
        s = sub[sub["wavelength"] == wl].set_index("geomag").reindex(geomag_order)
        x = [geomag_x[g] + (WAVELENGTHS.index(wl) - 3) * 0.03 for g in geomag_order]
        ax.plot(x, s["peak_lag_hours"], "o-", ms=7, lw=1.2, color=COLORS[wl],
                alpha=0.85, label=WL_LABEL[wl])
    ax.set_xticks(list(geomag_x.values()))
    ax.set_xticklabels(geomag_labels, fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_SUBTITLE)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.4, 2.4)
    ax.tick_params(labelsize=FS_TICK)

axes[0].set_ylabel("peak lag [h]", fontsize=FS_LABEL)
axes[1].legend(fontsize=FS_LEGEND, loc="best", ncol=2)
fig.suptitle("Peak lag by geomagnetic condition (full period)\n"
             "peak shifting between all/quiet would indicate storm-time "
             "contamination (TODO #8)", fontsize=FS_TITLE)
fig.tight_layout(rect=(0, 0, 1, 0.86))

OUT3 = OUT_DIR / "euv_peak_quiet_vs_all"
fig.savefig(f"{OUT3}.pdf")
fig.savefig(f"{OUT3}.png", dpi=160)
print(f"wrote {OUT3}.pdf / .png")
plt.close(fig)

# ===== ブートストラップCIの参考出力 (コンソールのみ、図には含めない) =====
print("\n=== bootstrap peak-lag CI summary (full period) ===")
print(boot.sort_values(["window", "geomag", "wavelength"]).to_string(index=False))
