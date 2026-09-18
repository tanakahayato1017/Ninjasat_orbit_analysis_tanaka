"""TLEエポック選択(日次/週次/月次)感度分析の可視化 (TODO #3, R2 Major#1/Minor#2)。

入力:
  data/figure_data/tle_epoch_sensitivity.csv  (13_tle_epoch_sensitivity.py の出力,
      mode x wavelength x lag のSpearmanラグ相関曲線)
  data/figure_data/tle_epoch_peaks.csv        (同, モード別ピークラグ表)
  data/figure_data/tle_epoch_drift.csv        (同, SGP4伝播誤差成長: monthly TLEの
      h_TLEとdaily TLE基準h_TLEとの差, days-since-epochでビン平均)
  data/figure_data/tle_epoch_modediff_stats.csv (同, 誤差成長率などのスカラー統計)

出力:
  figures/tle_epoch_sensitivity.pdf / .png
      左: モード別(daily/weekly/monthly)xwavelength別(304/1216nm)のラグ相関曲線。
          markerでピークラグ位置を強調。
      右: SGP4伝播誤差成長 (days-since-epoch vs h_TLE_monthly - h_TLE_daily,
          日次ビン中央値±SEM)。|diff|のabs(days)に対する線形フィット線を重ね描き、
          誤差成長率[m/day]を注記。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_tle_epoch_sensitivity.py

注意: figure_data中のwavelength列はCSV往復で整数型として読み込まれるため
(docs/2026-07-02_euv_lag_correlation.md で既知のピットフォール)、文字列化してから
比較する。
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CORR = ROOT / "data" / "figure_data" / "tle_epoch_sensitivity.csv"
IN_PEAKS = ROOT / "data" / "figure_data" / "tle_epoch_peaks.csv"
IN_DRIFT = ROOT / "data" / "figure_data" / "tle_epoch_drift.csv"
IN_STATS = ROOT / "data" / "figure_data" / "tle_epoch_modediff_stats.csv"
OUT_DIR = ROOT / "figures"

MODES = ["daily", "weekly", "monthly"]
WAVELENGTHS = ["304", "1216"]
MODE_COLORS = {"daily": "#2171b5", "weekly": "#238b45", "monthly": "#d62728"}
WL_LINESTYLE = {"304": "-", "1216": "--"}
WL_LABEL = {"304": "304 nm (EUV)", "1216": "1216 nm (Ly-alpha, FUV)"}

corr = pd.read_csv(IN_CORR)
peaks = pd.read_csv(IN_PEAKS)
drift = pd.read_csv(IN_DRIFT)
stats = pd.read_csv(IN_STATS)

# wavelength列はCSV往復で整数型になるため文字列化してから比較する
corr["wavelength"] = corr["wavelength"].astype(str)
peaks["wavelength"] = peaks["wavelength"].astype(str)

FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 13
FS_TITLE = 15
FS_SUBTITLE = 14
FS_TEXT = 12
FS_PANEL = 16

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ============================== 左: ラグ相関曲線 ==============================
ax = axes[0]
for mode in MODES:
    for wl in WAVELENGTHS:
        sub = corr[(corr["mode"] == mode) & (corr["wavelength"] == wl)].sort_values("lag_hours")
        ax.plot(sub["lag_hours"], sub["spearman_r"], WL_LINESTYLE[wl],
                 color=MODE_COLORS[mode], lw=1.7, alpha=0.85,
                 label=f"{mode}, {WL_LABEL[wl]}")

        prow = peaks[(peaks["mode"] == mode) & (peaks["wavelength"] == wl)]
        if len(prow) and np.isfinite(prow["peak_lag_hours"].iloc[0]):
            ax.plot(prow["peak_lag_hours"].iloc[0], prow["peak_r"].iloc[0], "o",
                     color=MODE_COLORS[mode], ms=8, mec="black", mew=0.7, zorder=5)

ax.axhline(0, color="gray", lw=0.6)
ax.axvline(6, color="gray", lw=0.6, ls=":")
ax.set_xlim(0, 120)
ax.set_xlabel("lag L [h]  (EUV(t-L) vs d(Delta h)/dt(t))", fontsize=FS_LABEL)
ax.set_ylabel("Spearman r", fontsize=FS_LABEL)
ax.set_title("Lag correlation by TLE selection mode\n"
              "(dots = peak, lag>=6h; SG window=13, old-paper style)", fontsize=FS_SUBTITLE)
ax.legend(fontsize=FS_LEGEND - 2, loc="upper left", ncol=1)
ax.grid(alpha=0.3)
ax.tick_params(labelsize=FS_TICK)
ax.text(0.02, 0.04, "(a)", transform=ax.transAxes, fontsize=FS_PANEL,
         fontweight="bold", va="bottom", ha="left")

# ========================== 右: SGP4伝播誤差成長 ==========================
ax2 = axes[1]
drift = drift.sort_values("days_since_epoch")
sem = drift["std_diff_m"] / np.sqrt(drift["n"].clip(lower=1))
ax2.errorbar(drift["days_since_epoch"], drift["median_diff_m"], yerr=sem,
              fmt="o-", color="#d62728", ms=5, lw=1.4, capsize=2.5,
              label="median(h_TLE,monthly - h_TLE,daily) +/- SEM")
ax2.axhline(0, color="gray", lw=0.6)
ax2.axvline(0, color="gray", lw=0.6, ls=":")

growth_rate = stats.loc[stats["quantity"] == "sgp4_error_growth_rate", "value"].iloc[0]
ax2.text(0.03, 0.03,
          f"|diff| growth rate (linear fit of\n"
          f"median|diff| vs |days|, |days|>=1):\n"
          f"{growth_rate:.1f} m/day",
          transform=ax2.transAxes, fontsize=FS_TEXT, va="bottom", ha="left",
          bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

ax2.set_xlabel("days since monthly-TLE epoch", fontsize=FS_LABEL)
ax2.set_ylabel("h_TLE,monthly - h_TLE,daily  [m]", fontsize=FS_LABEL)
ax2.set_title("SGP4 propagation error growth\n"
               "(monthly TLE vs. daily-refreshed TLE reference)", fontsize=FS_SUBTITLE)
ax2.legend(fontsize=FS_LEGEND, loc="upper right")
ax2.grid(alpha=0.3)
ax2.tick_params(labelsize=FS_TICK)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=FS_PANEL,
          fontweight="bold", va="top", ha="left")

fig.suptitle("R2 Major#1/Minor#2: TLE epoch selection sensitivity of the old "
              "(Delta h = TLE - GPS) proxy", fontsize=FS_TITLE)
fig.tight_layout(rect=(0, 0, 1, 0.90))

OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "tle_epoch_sensitivity"
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
plt.close(fig)

print("\n=== peak lag table ===")
print(peaks.sort_values(["wavelength", "mode"]).to_string(index=False))
