"""全期間(2024-04〜11)の dE/dt (E_J2ベース軌道減衰指標) 時系列図。

対応TODO: #1, #10 (3ヶ月分のみだった時系列図を全期間に拡張)

入力:  data/figure_data/dEdt_timeseries_full.csv  (03_dEdt_proxy.py の出力)
出力:  figures/dEdt_timeseries.pdf / .png

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行:  .venv/Scripts/python plots/plot_dEdt_timeseries.py

備考: 生の3hビン中心差分は点ごとのノイズが大きいため、視認性のため1日窓
(8点, 3h*8=24h) の単純移動平均を薄い実線として重ねる。これは表示上の補助線
であり、新たな統計量をfigure_data外で計算するものではない
(値そのものはCSVのdadt_m_per_dayの単純移動平均)。
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "dEdt_timeseries_full.csv"
OUT = ROOT / "figures" / "dEdt_timeseries"

ROLL_WINDOW = 8  # 1日 = 8 * 3h (表示用移動平均のみ)
YLIM_PCTL = (1, 99)  # 表示用の外れ値クリップ (生の3hビン中心差分は少数の外れ値を含む)

df = pd.read_csv(IN, parse_dates=["datetime"])
df = df.sort_values("datetime")
df["dadt_roll"] = df["dadt_m_per_day"].rolling(ROLL_WINDOW, center=True, min_periods=3).mean()
df["dEdt_roll"] = df["dEdt_Jkg_per_day"].rolling(ROLL_WINDOW, center=True, min_periods=3).mean()

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax = axes[0]
ax.axhline(0, color="gray", lw=0.6)
ax.plot(df["datetime"], df["dEdt_Jkg_per_day"] / 1e3, ".", ms=2, color="#1f77b4", alpha=0.25,
        label="3h bin, raw central diff")
ax.plot(df["datetime"], df["dEdt_roll"] / 1e3, "-", lw=1.2, color="#08306b",
        label=f"{ROLL_WINDOW*3}h moving average (display only)")
ax.set_ylabel(r"$dE_{J2}/dt$ [kJ kg$^{-1}$ day$^{-1}$]")
ax.set_title("NinjaSat orbital decay proxy: $dE/dt$ (J2-corrected specific energy), 2024-04 to 2024-11")
ax.legend(loc="lower left", fontsize=8)
# 欠測隣接ビンのdE/dtはNaN (03の欠測跨ぎ対策) のためnanpercentileを使う
lo, hi = np.nanpercentile(df["dEdt_Jkg_per_day"] / 1e3, YLIM_PCTL)
pad = 0.25 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)

ax = axes[1]
ax.axhline(0, color="gray", lw=0.6)
ax.plot(df["datetime"], df["dadt_m_per_day"], ".", ms=2, color="#d62728", alpha=0.25,
        label="3h bin, raw central diff")
ax.plot(df["datetime"], df["dadt_roll"], "-", lw=1.2, color="#7f0000",
        label=f"{ROLL_WINDOW*3}h moving average (display only)")
ax.set_ylabel(r"equivalent $da/dt$ [m day$^{-1}$]")
ax.set_xlabel("UTC")
ax.legend(loc="lower left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
lo, hi = np.nanpercentile(df["dadt_m_per_day"], YLIM_PCTL)
pad = 0.25 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)

n_out_e = int(((df["dEdt_Jkg_per_day"] / 1e3 < axes[0].get_ylim()[0]) |
               (df["dEdt_Jkg_per_day"] / 1e3 > axes[0].get_ylim()[1])).sum())
n_out_a = int(((df["dadt_m_per_day"] < ax.get_ylim()[0]) |
               (df["dadt_m_per_day"] > ax.get_ylim()[1])).sum())
fig.text(0.99, 0.01,
          f"note: y-axis clipped to {YLIM_PCTL[0]}-{YLIM_PCTL[1]} pctl "
          f"(+/-25% pad); {n_out_e} / {n_out_a} of {len(df)} raw points "
          "fall outside the E/a panels respectively (unsmoothed 3h central diff "
          "is noisy; see docs)",
          ha="right", va="bottom", fontsize=6.5, color="gray")

for ax in axes:
    ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
