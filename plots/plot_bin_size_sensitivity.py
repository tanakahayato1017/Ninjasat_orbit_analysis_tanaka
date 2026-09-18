"""ビン幅感度: 独立密度相関とEUVラグ相関ピークのビン幅依存性サマリ。

対応TODO: #9 (3時間サンプリング間隔の選定理由, R2)

入力:
  data/figure_data/bin_size_sensitivity.csv  (15_bin_size_sensitivity.py の出力,
    long形式: metric列で density_corr_daily / euv_lag_peak を区別)

出力: figures/bin_size_sensitivity.png / .pdf (1x3パネル、bin幅は共有x軸)
  (a) (-dE/dt)日平均 vs Swarm_B/GRACE-FO日平均密度のSpearman相関 vs ビン幅
  (b) irr_304ラグ相関ピークのSpearman r vs ビン幅
  (c) irr_304ラグ相関ピークラグ [h] vs ビン幅
  95分(1軌道)〜6時間 (レンジ約3.8倍) にわたって符号・強さ・ピーク位置が
  安定していることが一目で分かる形。1つの軸に相関係数とラグ[h]を混在させる
  二軸グラフは可読性を損なうため使わず、パネルを分けている。x軸は190分(2軌道)と
  3h(現行=180分)の実時間がほぼ等しく対数/線形の実測軸ではラベルが重なるため、
  4種類のビン幅を等間隔の順序カテゴリ (95min<190min<3h<6h) として並べる
  (順序のみを表現し、実時間間隔は非線形)。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_bin_size_sensitivity.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "figure_data" / "bin_size_sensitivity.csv"
OUT = ROOT / "figures" / "bin_size_sensitivity"

BIN_ORDER = ["95min", "190min", "3h", "6h"]
BIN_MINUTES = {"95min": 95.0, "190min": 190.0, "3h": 180.0, "6h": 360.0}
BIN_TICK_LABEL = {"95min": "95min\n(1 orbit)", "190min": "190min\n(2 orbits)",
                   "3h": "3h\n(current)", "6h": "6h"}
SAT_COLOR = {"Swarm_B": "#DD8452", "GRACE-FO": "#4C72B0"}

df = pd.read_csv(IN_CSV)
# 190min(2軌道)と3h(=180min,現行)は実時間がほぼ等しく、対数/線形の実測軸では
# 目盛りラベルが重なって読めなくなるため、順序だけを表す等間隔カテゴリ軸を使う。
x_all = np.arange(len(BIN_ORDER), dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

# ---------------- (a) 独立密度相関 vs ビン幅 ----------------
ax = axes[0]
dens = df[df["metric"] == "density_corr_daily"]
for sat, color in SAT_COLOR.items():
    s = dens[dens["satellite"] == sat].set_index("bin_width").reindex(BIN_ORDER)
    ax.plot(x_all, s["r"], "-o", ms=7, lw=1.4, color=color, label=sat)
ax.set_xticks(x_all)
ax.set_xticklabels([BIN_TICK_LABEL[b] for b in BIN_ORDER])
ax.set_xlim(-0.4, len(BIN_ORDER) - 0.6)
ax.set_ylabel(r"Spearman $\rho$  ($-dE/dt$ daily vs density daily)")
ax.set_title("(a) independent density correlation", fontsize=10)
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3)

# ---------------- (b) EUVラグ相関ピークr vs ビン幅 ----------------
ax = axes[1]
euv = df[df["metric"] == "euv_lag_peak"].set_index("bin_width").reindex(BIN_ORDER)
ax.plot(x_all, euv["r"], "-o", ms=7, lw=1.4, color="#7f0000")
ax.set_xticks(x_all)
ax.set_xticklabels([BIN_TICK_LABEL[b] for b in BIN_ORDER])
ax.set_xlim(-0.4, len(BIN_ORDER) - 0.6)
ax.set_ylabel(r"peak Spearman $r$  (proxy vs irr\_304, lag$\geq$6h)")
ax.set_title("(b) EUV(304nm) lag-correlation peak strength", fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.grid(alpha=0.3)

# ---------------- (c) EUVラグ相関ピークラグ vs ビン幅 ----------------
ax = axes[2]
ax.plot(x_all, euv["peak_lag_hours"], "-o", ms=7, lw=1.4, color="#2171b5")
for xi, b in zip(x_all, BIN_ORDER):
    ax.annotate(f"{euv.loc[b, 'peak_lag_hours']:.0f}h", xy=(xi, euv.loc[b, "peak_lag_hours"]),
                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(x_all)
ax.set_xticklabels([BIN_TICK_LABEL[b] for b in BIN_ORDER])
ax.set_xlim(-0.4, len(BIN_ORDER) - 0.6)
ax.set_ylabel("peak lag [h]")
ax.set_title("(c) EUV(304nm) lag-correlation peak position", fontsize=10)
ax.set_ylim(30, 65)
ax.grid(alpha=0.3)

fig.suptitle(
    "Bin-width sensitivity: 95min (1 orbit) to 6h ($\\approx$3.8x range) — sign, strength, "
    "and peak-lag conclusions are stable across the full range", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
