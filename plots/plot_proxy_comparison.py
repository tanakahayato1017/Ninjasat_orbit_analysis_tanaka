"""新旧の軌道減衰proxyの比較図 (時系列の重ね描き + 散布図)。

対応TODO: #1, #4 (感度分析の可視化)

入力:  data/figure_data/proxy_comparison.csv        (04_compare_old_proxy.py の出力,
         --window未指定=平滑化なしの新旧proxy時系列)
       data/figure_data/proxy_comparison_stats.csv  (--window ごとの相関係数)
出力:  figures/proxy_comparison.pdf / .png

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行:  .venv/Scripts/python plots/plot_proxy_comparison.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "proxy_comparison.csv"
IN_STATS = ROOT / "data" / "figure_data" / "proxy_comparison_stats.csv"
OUT = ROOT / "figures" / "proxy_comparison"

YLIM_PCTL = (1, 99)  # 表示用の外れ値クリップ (生の3hビン中心差分は少数の外れ値を含む)

df = pd.read_csv(IN, parse_dates=["datetime"]).sort_values("datetime")
stats = pd.read_csv(IN_STATS)
row_raw = stats[stats["window"] == -1].iloc[0]

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

# --- 上段: 時系列を2軸で重ね描き ---
ax1 = fig.add_subplot(gs[0, :])
ax1.axhline(0, color="gray", lw=0.6)
l1, = ax1.plot(df["datetime"], df["dadt_new_m_per_day"], "-", lw=0.8, color="#d62728",
                alpha=0.8, label=r"new: equivalent $da/dt$ ($E_{J2}$-based) [m day$^{-1}$]")
ax1.set_ylabel(r"new proxy: $da/dt$ [m day$^{-1}$]", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title("New ($E_{J2}$-based) vs old (TLE$-$GPS altitude difference) decay proxy, "
              "3h bins, no smoothing")

ax2 = ax1.twinx()
l2, = ax2.plot(df["datetime"], df["ddeltahdt_old_m_per_day"], "-", lw=0.8, color="#1f77b4",
                alpha=0.6, label=r"old: $d(\Delta h)/dt$ (TLE$-$GPS) [m day$^{-1}$]")
ax2.set_ylabel(r"old proxy: $d(\Delta h)/dt$ [m day$^{-1}$]", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")

ax1.legend(handles=[l1, l2], loc="upper right", fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(alpha=0.3)

# 欠測隣接ビンの微分値はNaN (04の欠測跨ぎ対策) のためnanpercentileを使う
lo1, hi1 = np.nanpercentile(df["dadt_new_m_per_day"], YLIM_PCTL)
pad1 = 0.4 * (hi1 - lo1)
ax1.set_ylim(lo1 - pad1, hi1 + pad1)
lo2, hi2 = np.nanpercentile(df["ddeltahdt_old_m_per_day"], YLIM_PCTL)
pad2 = 0.4 * (hi2 - lo2)
ax2.set_ylim(lo2 - pad2, hi2 + pad2)
fig.text(0.01, 0.505,
          f"note: y-axes clipped to {YLIM_PCTL[0]}-{YLIM_PCTL[1]} pctl (+/-40% pad) "
          "for readability; unsmoothed 3h central diff is noisy (see docs)",
          ha="left", va="top", fontsize=6.5, color="gray")

# --- 左下: 散布図 (平滑化なし) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df["dadt_new_m_per_day"], df["ddeltahdt_old_m_per_day"], s=6, alpha=0.3,
            color="#333333")
ax3.set_xlabel(r"new: $da/dt$ [m day$^{-1}$]")
ax3.set_ylabel(r"old: $d(\Delta h)/dt$ [m day$^{-1}$]")
ax3.set_title(f"no smoothing: Pearson r={row_raw['pearson_r']:+.3f}, "
              f"Spearman ρ={row_raw['spearman_r']:+.3f}\n(n={int(row_raw['n'])})",
              fontsize=9)
ax3.grid(alpha=0.3)
ax3.axhline(0, color="gray", lw=0.5)
ax3.axvline(0, color="gray", lw=0.5)
xlo, xhi = np.nanpercentile(df["dadt_new_m_per_day"], YLIM_PCTL)
ylo, yhi = np.nanpercentile(df["ddeltahdt_old_m_per_day"], YLIM_PCTL)
xpad, ypad = 0.5 * (xhi - xlo), 0.5 * (yhi - ylo)
ax3.set_xlim(xlo - xpad, xhi + xpad)
ax3.set_ylim(ylo - ypad, yhi + ypad)

# --- 右下: SGフィルタ窓幅に対する相関係数の感度 (TODO #4) ---
ax4 = fig.add_subplot(gs[1, 1])
swept = stats[stats["window"] != -1].sort_values("window")
ax4.axhline(0, color="gray", lw=0.6)
ax4.plot(swept["window"], swept["pearson_r"], "o-", color="#2ca02c", label="Pearson r")
ax4.plot(swept["window"], swept["spearman_r"], "s-", color="#9467bd", label="Spearman ρ")
ax4.scatter([0], [row_raw["pearson_r"]], color="#2ca02c", marker="x", s=50,
            label="Pearson r (no smoothing)")
ax4.set_xlabel("Savitzky-Golay window [points] (0 = no smoothing)")
ax4.set_ylabel("correlation coefficient")
ax4.set_title("sensitivity to SG smoothing window (TODO #4)", fontsize=9)
ax4.legend(fontsize=7, loc="best")
ax4.grid(alpha=0.3)

fig.tight_layout()

OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
