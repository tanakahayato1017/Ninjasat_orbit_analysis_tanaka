"""旧解析(MT_code)のGNSS-defined altitude定義の逆同定・不整合項の可視化。

対応TODO: #1, #2, #12 ("GNSS-defined altitude"の定義を明記, R1指摘)

入力:
  data/figure_data/old_altitude_definition_audit.csv
      12_old_altitude_definition_audit.py の出力。ビンごとの候補再構成値と
      旧ファイル値、α(ジオイド項)・Δh_old・Δh_degeoidedのビン平均。
  data/figure_data/old_altitude_definition_geometric_2day.csv
      同スクリプトの出力。1周回スケール可視化用の生カデンス(~600s)2日分。

出力: figures/old_altitude_audit.png / .pdf

レイアウト:
  上段左: GPS候補(a,b,c,c')の再構成値 vs 旧ファイルgps_average_altitude散布図 (RMS注記)
  上段右: TLE候補(i,ii)の再構成値 vs 旧ファイルtle_average_altitude散布図 (RMS注記)
  下段  : 不整合項の時系列 (Δh_old, α[ジオイド項], Δh_degeoided) を
          1周回スケールが見える2日分で拡大表示

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行: .venv/Scripts/python plots/plot_old_altitude_audit.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_BINS = ROOT / "data" / "figure_data" / "old_altitude_definition_audit.csv"
IN_2DAY = ROOT / "data" / "figure_data" / "old_altitude_definition_geometric_2day.csv"
OUT = ROOT / "figures" / "old_altitude_audit"

# 固定順の色割当て (repo内の既存図と揃えた配色: blue/red/green/orange/purple系)
GPS_COLORS = {
    "gps_a_msl_raw": "#4C72B0",                      # 青: (a) 生MSL高度
    "gps_b_ellipsoidal_h_ell": "#DD8452",            # 橙: (b) WGS84楕円体高
    "gps_c_r_minus_R0_geoid_corrected": "#55A868",   # 緑: (c) r-R0 (ジオイド補正)
    "gps_cprime_r_minus_R0_asused": "#C44E52",       # 赤: (c') r-R0 (旧コード実装)
}
GPS_LABELS = {
    "gps_a_msl_raw": "(a) mean(gpsAltitudeMeters)",
    "gps_b_ellipsoidal_h_ell": "(b) mean(alt+geoid) = h_ell",
    "gps_c_r_minus_R0_geoid_corrected": "(c) |ECEF(h_ell)| - R0",
    "gps_cprime_r_minus_R0_asused": "(c') |ECEF(raw alt)| - R0  [as coded in MT_code]",
}
TLE_COLORS = {
    "tle_i_r_minus_R0_asused": "#C44E52",            # 赤: (i) 旧コード実装
    "tle_ii_geodetic_height_correct": "#8172B2",     # 紫: (ii) 正しい測地高
}
TLE_LABELS = {
    "tle_i_r_minus_R0_asused": "(i) |r_SGP4| - R0  [as coded in MT_code]",
    "tle_ii_geodetic_height_correct": "(ii) SGP4 geodetic height (correct def.)",
}

bins = pd.read_csv(IN_BINS, parse_dates=["timestamp"])
two_day = pd.read_csv(IN_2DAY, parse_dates=["datetime"])

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.28)

# --- 上段左: GPS候補の残差散布図 (candidate - archive, Bland-Altman式) ---
# 候補(a)/(b)は候補(c)/(c')よりスケールが~11km違うため、生値どうしのy=xプロットでは
# (a)/(b)がほぼ視野外になり比較にならない。残差(縦軸)で揃えて全候補を同一パネルで
# 比較できるようにする。
ax_gps = fig.add_subplot(gs[0, 0])
old_gps = bins["old_gps_average_altitude"].to_numpy()
ax_gps.axhline(0, color="0.35", lw=1, ls="--", zorder=0)
for col, color in GPS_COLORS.items():
    y = bins[col].to_numpy()
    ok = np.isfinite(y) & np.isfinite(old_gps)
    resid_m = (y[ok] - old_gps[ok]) * 1000
    rms_m = np.sqrt(np.mean(resid_m ** 2))
    ax_gps.scatter(old_gps[ok], resid_m, s=10, alpha=0.55, color=color, edgecolors="none",
                   label=f"{GPS_LABELS[col]}  (RMS={rms_m:.0f} m)")
ax_gps.set_xlabel("archived gps_average_altitude [km]")
ax_gps.set_ylabel("candidate - archive [m]")
ax_gps.set_yscale("symlog", linthresh=1000)
ax_gps.set_title("GPS-side candidates: residual vs archive (2024-07)")
ax_gps.legend(fontsize=7.5, loc="lower right")
ax_gps.grid(alpha=0.3)

# --- 上段右: TLE候補の残差散布図 ---
ax_tle = fig.add_subplot(gs[0, 1])
old_tle = bins["old_tle_average_altitude"].to_numpy()
ax_tle.axhline(0, color="0.35", lw=1, ls="--", zorder=0)
for col, color in TLE_COLORS.items():
    y = bins[col].to_numpy()
    ok = np.isfinite(y) & np.isfinite(old_tle)
    resid_m = (y[ok] - old_tle[ok]) * 1000
    rms_m = np.sqrt(np.mean(resid_m ** 2))
    ax_tle.scatter(old_tle[ok], resid_m, s=10, alpha=0.55, color=color, edgecolors="none",
                   label=f"{TLE_LABELS[col]}  (RMS={rms_m:.0f} m)")
ax_tle.set_xlabel("archived tle_average_altitude [km]")
ax_tle.set_ylabel("candidate - archive [m]")
ax_tle.set_yscale("symlog", linthresh=1000)
ax_tle.set_title("TLE-side candidates: residual vs archive (2024-07)")
ax_tle.legend(fontsize=8, loc="lower right")
ax_tle.grid(alpha=0.3)

# --- 下段: 不整合項の時系列 (1周回スケールが見える2日分) ---
ax_ts = fig.add_subplot(gs[1, :])
ax_ts.axhline(0, color="0.5", lw=0.7, zorder=0)
ax_ts.plot(two_day["datetime"], two_day["delta_h_old"] * 1000, color="#C44E52", lw=1.1,
           label=r"$\Delta h_{\mathrm{old}}$ = TLE(i) - GPS(c')  [actual old-analysis quantity]")
ax_ts.plot(two_day["datetime"], two_day["delta_h_degeoided"] * 1000, color="#8172B2", lw=1.1,
           alpha=0.85,
           label=r"$\Delta h_{\mathrm{degeoided}}$ = TLE(i) - GPS(c)  [geoid term removed]")
ax_ts.plot(two_day["datetime"], two_day["alpha_geoid"] * 1000, color="#55A868", lw=1.3,
           label=r"$\alpha$ = GPS(c) - GPS(c')  [uncorrected-geoid offset]")
ax_ts.set_xlabel("Time [UTC]")
ax_ts.set_ylabel("altitude difference [m]")
ax_ts.set_title("Inconsistency terms, zoomed to single-orbit scale (~94.5 min); 2024-07-10 to 07-12")
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax_ts.tick_params(axis="x", rotation=30)
ax_ts.legend(fontsize=8, loc="upper right")
ax_ts.grid(alpha=0.3)

fig.suptitle('Reverse-engineered definition of the old "gps/tle_average_altitude" and its inconsistency terms',
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=150)
fig.savefig(OUT.with_suffix(".pdf"))
print(f"wrote {OUT.with_suffix('.png')}")
print(f"wrote {OUT.with_suffix('.pdf')}")
