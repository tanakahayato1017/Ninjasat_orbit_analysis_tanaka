"""高度解釈の判定実験 (図2: altitude_semantics)。

対応TODO: #1, #12 ("GNSS-defined altitude"の定義を明記, R1指摘)

入力:
  data/figure_data/altitude_semantics.csv        (H1/H2/H3残差の時系列, 緯度付き)
  data/figure_data/altitude_semantics_stats.csv  (bias/std)
出力: figures/altitude_semantics.png / .pdf

レイアウト:
  上段: H1 (楕円体高解釈) の残差 [m]。緯度で色付けした散布+細線。
  下段: H2 (alt+Req)・H3 (alt+Rmean) の残差 [km]。同じ緯度カラーマップを共有し、
        2曲線を重ねて描く (形はほぼ同じ、定数分だけ縦にずれる)。
  各パネルにbias/stdを注記。

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行: .venv/Scripts/python plots/plot_altitude_semantics.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_TS = ROOT / "data" / "figure_data" / "altitude_semantics.csv"
IN_STATS = ROOT / "data" / "figure_data" / "altitude_semantics_stats.csv"
OUT = ROOT / "figures" / "altitude_semantics"

FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 13
FS_TITLE = 16
FS_SUBTITLE = 14
FS_PANEL = 16

df = pd.read_csv(IN_TS, parse_dates=["datetime"])
stats = pd.read_csv(IN_STATS).set_index("hypothesis")

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# --- 上段: H1 (採用した楕円体高解釈) ---
ax = axes[0]
ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)
ax.plot(df["datetime"], df["resid_H1_m"], "-", lw=0.8, color="0.75", zorder=1)
sc = ax.scatter(df["datetime"], df["resid_H1_m"], c=df["lat_deg"], cmap="coolwarm",
                 s=22, vmin=-90, vmax=90, zorder=2)
s1 = stats.loc["H1_ellipsoidal"]
ax.set_ylabel(r"H1 residual: $|\mathrm{ECEF}(\phi,\lambda,\mathrm{alt+geoid})| - |\mathbf{r}_{\mathrm{SGP4}}|$"
              "\n[m]", fontsize=FS_LABEL - 3)
ax.set_title(f"(H1) ellipsoidal-height interpretation  "
             f"bias={s1['bias']:+.1f} m, std={s1['std']:.1f} m  (adopted)",
             fontsize=FS_SUBTITLE)
ax.grid(alpha=0.3)
ax.tick_params(labelsize=FS_TICK)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("GPS latitude [deg]", fontsize=FS_LABEL - 3)
cb.ax.tick_params(labelsize=FS_TICK)
ax.text(0.98, 0.94, "(a)", transform=ax.transAxes, fontsize=FS_PANEL,
         fontweight="bold", va="top", ha="right")

# --- 下段: H2 / H3 (素朴な球面解釈) ---
ax = axes[1]
ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)
s2 = stats.loc["H2_alt_plus_Req"]
s3 = stats.loc["H3_alt_plus_Rmean"]
ax.plot(df["datetime"], df["resid_H2_km"], "-", lw=1.1, color="#C44E52",
        label=fr"(H2) alt + $R_{{eq}}$ (6378.137 km)   bias={s2['bias']:+.2f} km, std={s2['std']:.2f} km")
ax.plot(df["datetime"], df["resid_H3_km"], "-", lw=1.1, color="#8172B2",
        label=fr"(H3) alt + $R_{{mean}}$ (6371 km)   bias={s3['bias']:+.2f} km, std={s3['std']:.2f} km")
sc2 = ax.scatter(df["datetime"], df["resid_H2_km"], c=df["lat_deg"], cmap="coolwarm",
                  s=16, vmin=-90, vmax=90, zorder=3)
ax.set_ylabel("H2 / H3 residual: alt$+$const $-\\ |\\mathbf{r}_{\\mathrm{SGP4}}|$\n[km]",
              fontsize=FS_LABEL - 3)
ax.set_xlabel("UTC, 2024-07-01", fontsize=FS_LABEL)
ax.set_title("(H2/H3) naive spherical interpretations of gpsAltitudeMeters (rejected)",
             fontsize=FS_SUBTITLE)
ax.legend(fontsize=FS_LEGEND, loc="upper right")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.tick_params(labelsize=FS_TICK)
ax.text(0.005, 0.94, "(b)", transform=ax.transAxes, fontsize=FS_PANEL,
         fontweight="bold", va="top", ha="left")

fig.suptitle("Altitude-interpretation hypothesis test: GNSS-derived $|\\mathbf{r}|$ vs. SGP4 $|\\mathbf{r}|$,\n"
             "2024-07-01 (nearest TLE epoch 2024-07-01 04:37 UTC)", fontsize=FS_TITLE, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.92))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
fig.savefig(OUT.with_suffix(".pdf"))
print(f"wrote {OUT.with_suffix('.png')}")
print(f"wrote {OUT.with_suffix('.pdf')}")
