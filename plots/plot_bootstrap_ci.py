"""ブロックブートストラップCI (ピークラグ) の区間プロット。

対応TODO: #1, #4, #6, #8 (不確かさの定量化、ピークラグの信頼区間)

入力:
  data/figure_data/euv_lag_bootstrap.csv  (10_euv_lag_correlation.py の出力,
    5日非重複ブロックのbootstrap 500回、percentile 2.5/16/50/84/97.5)

出力: figures/bootstrap_peak_lag.png / .pdf
  window=13 (SG~39h) x geomag{all, quiet_kp4} の2パネル。波長を縦に並べ、
  p2.5-p97.5を細線、p16-p84を太線、p50を点で表示。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_bootstrap_ci.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_BOOT = ROOT / "data" / "figure_data" / "euv_lag_bootstrap.csv"
OUT = ROOT / "figures" / "bootstrap_peak_lag"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
EUV_BAND = ["256", "284", "304"]
FUV_BAND = ["1175", "1216", "1335", "1405"]
EUV_COLORS = dict(zip(EUV_BAND, ["#7f0000", "#d62728", "#ff9896"]))
FUV_COLORS = dict(zip(FUV_BAND, ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]))
COLORS = {**EUV_COLORS, **FUV_COLORS}
WL_LABEL = {"256": "25.6 nm", "284": "28.4 nm", "304": "30.4 nm",
            "1175": "117.5 nm", "1216": "121.6 nm (Ly-a)",
            "1335": "133.5 nm", "1405": "140.5 nm"}

FS_LABEL = 17
FS_TICK = 14
FS_TITLE = 15
FS_SUBTITLE = 14
FS_PANEL = 16

boot = pd.read_csv(IN_BOOT)
boot["wavelength"] = boot["wavelength"].astype(str)
boot13 = boot[boot["window"] == 13]

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True, sharex=True,
                          layout="constrained")

panel_labels = ["(a)", "(b)"]
for ax, geomag, title, lab in zip(
        axes, ["all", "quiet_kp4"],
        ["geomag = all", "geomag = quiet ($\\mathrm{Kp}^{max}_{48h}$<4)"],
        panel_labels):
    sub = boot13[boot13["geomag"] == geomag].set_index("wavelength").reindex(WAVELENGTHS)
    y = range(len(WAVELENGTHS))
    for yi, wl in zip(y, WAVELENGTHS):
        row = sub.loc[wl]
        c = COLORS[wl]
        ax.plot([row["p2_5"], row["p97_5"]], [yi, yi], "-", lw=1.6, color=c,
                alpha=0.6, solid_capstyle="round")
        ax.plot([row["p16"], row["p84"]], [yi, yi], "-", lw=7.5, color=c,
                alpha=0.85, solid_capstyle="round")
        ax.plot([row["p50"]], [yi], "o", ms=7, mfc="white", mec=c, mew=1.8,
                zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([WL_LABEL[wl] for wl in WAVELENGTHS], fontsize=FS_TICK)
    ax.set_xlabel("bootstrap peak-lag distribution [h]", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_SUBTITLE)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 120)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

axes[0].set_ylim(len(WAVELENGTHS) - 0.3, -1.1)

fig.suptitle(
    "Block-bootstrap 95% CI of the peak lag (5-day blocks, 500 resamples, SG window=13)\n"
    "thin line = p2.5-p97.5 (95% CI), thick line = p16-p84 (68% CI), dot = median",
    fontsize=FS_TITLE)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
