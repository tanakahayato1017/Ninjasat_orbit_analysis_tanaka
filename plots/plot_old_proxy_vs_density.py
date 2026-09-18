"""旧proxy d(Δh)/dt (日次差分) と独立熱圏密度データの整合性チェック図。

対応TODO: #6 (旧手法の位置づけ), R2 Major#1関連

入力:  data/figure_data/old_proxy_vs_density.csv        (20_old_proxy_density_check.py
         の出力, 衛星別・日次ペア)
       data/figure_data/old_proxy_vs_density_stats.csv  (同スクリプトの出力,
         衛星別Pearson/Spearman + 新proxyの対応する相関値の参考注記)
出力:  figures/old_proxy_vs_density.pdf / .png

レイアウト:
  3パネル散布図 (Swarm A, Swarm B, GRACE-FO)。x軸=旧proxy d(altitude_diff)/dt
  [m/day]、y軸=衛星別日平均密度 [kg/m^3]。各パネルタイトルに旧proxyの
  Pearson/Spearman相関係数、右下に新proxy (-dE/dt) の対応する相関値
  (参考, dEdt_vs_density_stats.csv由来の既知値) を注記する。

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行:  .venv/Scripts/python plots/plot_old_proxy_vs_density.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "old_proxy_vs_density.csv"
IN_STATS = ROOT / "data" / "figure_data" / "old_proxy_vs_density_stats.csv"
OUT = ROOT / "figures" / "old_proxy_vs_density"

SATS = ["Swarm_A", "Swarm_B", "GRACE-FO"]
COLORS = {"Swarm_A": "#ff7f0e", "Swarm_B": "#1f77b4", "GRACE-FO": "#2ca02c"}
LABELS = {"Swarm_A": "Swarm A (~477 km, reference)",
          "Swarm_B": "Swarm B (~515 km)",
          "GRACE-FO": "GRACE-FO (~488 km)"}

df = pd.read_csv(IN, parse_dates=["date"])
stats = pd.read_csv(IN_STATS).set_index("satellite")

# x軸表示範囲: 2024-11-01 (GRACE-FO -2547 m/day 等) は月境界を跨ぐ日次差分で、
# 旧解析が月ごとに独立にTLEフィッティングしていることに由来する見かけの
# 大外れ値 (docs/2026-07-02_tle_epoch_sensitivity.md 参照)。相関係数の計算
# (src/20_old_proxy_density_check.py, Spearmanベース) には作用済み・変更しないが、
# 可視化ではこの1点で軸が潰れてしまうため、データの大部分 (1-99パーセンタイル)
# を基準にクリップし、範囲外の点数を注記する。
all_x = df["old_proxy_ddeltahdt_m_per_day"].to_numpy()
lo, hi = np.percentile(all_x, [1, 99])
pad = 0.15 * (hi - lo)
XLIM = (lo - pad, hi + pad)

FS_LABEL = 17
FS_TICK = 14
FS_TITLE = 15
FS_SUBTITLE = 13
FS_TEXT = 11
FS_PANEL = 16

fig, axes = plt.subplots(1, 3, figsize=(17, 8.0))

panel_labels = ["(a)", "(b)", "(c)"]
for ax, sat, lab in zip(axes, SATS, panel_labels):
    sub = df[df["satellite"] == sat]
    x = sub["old_proxy_ddeltahdt_m_per_day"]
    n_off = int(((x < XLIM[0]) | (x > XLIM[1])).sum())
    ax.scatter(x, sub["rho_mean"], s=12, alpha=0.55, color=COLORS[sat])
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"old proxy $d(\Delta h)/dt$ [m day$^{-1}$]" "\n" r"($\Delta h$ = TLE $-$ GPS)",
                  fontsize=FS_LABEL - 3)
    if sat == SATS[0]:
        ax.set_ylabel(r"density $\rho$ [kg m$^{-3}$]", fontsize=FS_LABEL)

    r = stats.loc[sat]
    ax.set_title(f"{LABELS[sat]}\n"
                 f"old proxy: Pearson r={r['pearson_r']:+.3f}\n"
                 f"Spearman ρ={r['spearman_r']:+.3f} (n={int(r['n'])})",
                 fontsize=FS_SUBTITLE - 1)
    ax.text(0.97, 0.04,
             f"new proxy $(-dE/dt)$\nSpearman ρ={r['new_proxy_spearman_r_ref']:+.3f} (ref)",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_TEXT,
             color="dimgray",
             bbox=dict(boxstyle="round", fc="white", ec="lightgray", alpha=0.85))
    if n_off:
        ax.text(0.97, 0.96, f"{n_off} pt off-scale\n(month-boundary jump,\nnot excluded from ρ)",
                 transform=ax.transAxes, ha="right", va="top", fontsize=FS_TEXT, color="firebrick")
    ax.axvline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.995, lab, transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

fig.suptitle("Old proxy consistency check: $d(\\Delta h)/dt$ (1-day difference, "
              "consecutive days only) vs independent thermosphere density\n"
              "(TU Delft portal), daily mean, 2024-04 to 2024-11",
              fontsize=FS_TITLE, y=0.985)

fig.text(0.01, 0.015,
          "note: old proxy = daily mean altitude_diff_km (>=4/8 3h-bins/day, src/20_old_proxy_density_check.py) "
          "differenced only across consecutive calendar days; positive correlation is physically expected\n"
          "(Delta h = TLE - GPS grows during decay); new-proxy values are reference figures from "
          "data/figure_data/dEdt_vs_density_stats.csv (remediated pipeline)",
          ha="left", va="bottom", fontsize=FS_TEXT, color="gray")

fig.subplots_adjust(left=0.06, right=0.98, bottom=0.21, top=0.78, wspace=0.32)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
