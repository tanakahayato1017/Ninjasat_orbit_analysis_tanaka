"""R1添付図の再現: NinjaSat楕円体高と比力学的エネルギーの軌道内変動比較。

対応TODO: #1

入力:  data/figure_data/energy_vs_altitude_conservation.csv
出力:  figures/energy_conservation.pdf / .png

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行:  .venv/Scripts/python plots/plot_energy_conservation.py
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "energy_vs_altitude_conservation.csv"
OUT = ROOT / "figures" / "energy_conservation"

FS_LABEL = 17
FS_TICK = 14
FS_TITLE = 14
FS_PANEL = 16

df = pd.read_csv(IN, parse_dates=["datetime"])

fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)

ax = axes[0]
ax.plot(df["datetime"], df["h_ell_m"] / 1e3, ".-", ms=3, lw=0.7, color="#333")
ax.set_ylabel("Ellipsoidal height\n[km]", fontsize=FS_LABEL)
ax.set_title("NinjaSat, 2024-07-01 to 2024-07-02 (GNSS telemetry, ~600 s sampling)",
             fontsize=FS_TITLE)

ax = axes[1]
e0 = df["E_point_Jkg"].mean()
ax.plot(df["datetime"], (df["E_point_Jkg"] - e0) / 1e3, ".-", ms=3, lw=0.7,
        color="#1f77b4")
ax.set_ylabel(r"$E$ (point mass) $-$ mean" + "\n[kJ kg$^{-1}$]", fontsize=FS_LABEL)

ax = axes[2]
e0 = df["E_J2_Jkg"].mean()
ax.plot(df["datetime"], (df["E_J2_Jkg"] - e0) / 1e3, ".-", ms=3, lw=0.7,
        color="#d62728")
ax.set_ylabel(r"$E$ (with $J_2$) $-$ mean" + "\n[kJ kg$^{-1}$]", fontsize=FS_LABEL)
ax.set_xlabel("UTC", fontsize=FS_LABEL)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

panel_labels = ["(a)", "(b)", "(c)"]
for ax, lab in zip(axes, panel_labels):
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.01, 0.94, lab, transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")
fig.autofmt_xdate()
fig.tight_layout()

OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
