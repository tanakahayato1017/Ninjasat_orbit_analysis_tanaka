"""新旧proxy相関の「符号反転」診断3枚組 (SGフィルタ非依存の3診断のまとめ図)。

対応TODO: #1, #4

入力:
  data/figure_data/sign_audit_lagdiff.csv       (08_sign_audit.py 診断1)
  data/figure_data/sign_audit_highpass_corr.csv (08_sign_audit.py 診断2)
  data/figure_data/sign_audit_daily_diff.csv    (16_diagnostics_figdata.py 診断3,
                                                  08と同一ロジックの再実装)

出力: figures/sign_audit_diagnostics.png / .pdf (1x3パネル)
  (a) ラグ差分相関 vs 時間スケール。符号反転点(6-12h)を縦線で強調。
  (b) 高周波残差 (24h移動平均除去) の相関行列ヒートマップ。
  (c) 日次ブロック平均の単純1日差分の散布図 (新proxy da/dt vs 旧proxy d(Delta h)/dt)。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_sign_audit.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_LAGDIFF = ROOT / "data" / "figure_data" / "sign_audit_lagdiff.csv"
IN_HIGHPASS = ROOT / "data" / "figure_data" / "sign_audit_highpass_corr.csv"
IN_DAILYDIFF = ROOT / "data" / "figure_data" / "sign_audit_daily_diff.csv"
OUT = ROOT / "figures" / "sign_audit_diagnostics"

lagdiff = pd.read_csv(IN_LAGDIFF)
highpass = pd.read_csv(IN_HIGHPASS, index_col=0)
dailydiff = pd.read_csv(IN_DAILYDIFF, parse_dates=["date"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ---------------- (a) ラグ差分相関 vs 時間スケール ----------------
ax = axes[0]
ax.axhline(0, color="gray", lw=0.7, zorder=0)
ax.axvspan(6, 12, color="#d62728", alpha=0.15, zorder=0, label="sign flip (6-12h)")
ax.plot(lagdiff["lag_hours"], lagdiff["spearman_r"], "-o", ms=3.5, lw=1.2,
        color="#1f77b4", label="Spearman r")
ax.plot(lagdiff["lag_hours"], lagdiff["pearson_r"], "-o", ms=3, lw=1.0,
        color="#9ecae1", alpha=0.85, label="Pearson r")
ax.set_xlabel("time scale $k$ [h]  ($x_{t+k}-x_t$ vs $y_{t+k}-y_t$)")
ax.set_ylabel("correlation")
ax.set_title("(a) lag-differenced correlation vs time scale\n"
             "(no SG filter; sign flips low-freq (+) $\\to$ high-freq ($-$))",
             fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 120)

# ---------------- (b) 高周波残差の相関行列ヒートマップ ----------------
ax = axes[1]
labels = ["E_J2", "gps_alt", "tle_alt", "delta_h"]
mat = highpass.loc[labels, labels].to_numpy()
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = mat[i, j]
        txt_color = "white" if abs(v) > 0.6 else "black"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                color=txt_color, fontsize=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.set_label("Spearman r", fontsize=8)
ax.set_title("(b) high-frequency residual correlation matrix\n"
             "(24h rolling-mean removed, 3h bins)", fontsize=9.5)

# ---------------- (c) 日次ブロック差分の散布図 ----------------
ax = axes[2]
x = dailydiff["dadt_m_per_day"].to_numpy()
y = dailydiff["ddhdt_m_per_day"].to_numpy()
rho, pval = spearmanr(x, y)
ax.axhline(0, color="gray", lw=0.6, zorder=0)
ax.axvline(0, color="gray", lw=0.6, zorder=0)
ax.scatter(x, y, s=16, alpha=0.55, color="#2ca02c", edgecolors="none")
ax.set_xlabel(r"new proxy $da/dt$ [m day$^{-1}$]  (from $E_{J2}$)")
ax.set_ylabel(r"old proxy $d(\Delta h)/dt$ [m day$^{-1}$]  ($\Delta h$=TLE$-$GPS)")
ax.set_title("(c) daily-block 1-day differences\n"
             f"(filter-independent; n={len(x)}, Spearman $\\rho$={rho:+.3f}, "
             f"p={pval:.1e})", fontsize=9.5)
ax.text(0.03, 0.05,
        "physical expectation: negative\n(both track orbital decay)",
        transform=ax.transAxes, fontsize=7.5, color="dimgray", va="bottom")
ax.grid(alpha=0.3)

fig.suptitle('Sign-audit of the new/old proxy correlation: 3 filter-independent diagnostics '
             '(src/08_sign_audit.py)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
