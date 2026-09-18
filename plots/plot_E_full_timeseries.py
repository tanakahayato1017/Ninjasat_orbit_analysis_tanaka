"""ミッション全期間のE_J2時系列 (図4: E_full_timeseries)。

対応TODO: #1 (軌道減衰指標の再検討) の全体像可視化

入力:
  data/figure_data/E_full_timeseries.csv                  (3hビン平均のE_J2/a_equiv)
  data/figure_data/E_full_timeseries_storm_intervals.csv   (Kp>=6の連続区間)
出力: figures/E_full_timeseries.png / .pdf

レイアウト: 1パネル。左軸 = E_J2 (3hビン平均, kJ/kg)、右軸 = 対応する等価半長軸
a_equiv (km)。Kp>=6の期間 (Gannon storm 5/10-12を含む) を縦帯で表示。

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行: .venv/Scripts/python plots/plot_E_full_timeseries.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_TS = ROOT / "data" / "figure_data" / "E_full_timeseries.csv"
IN_STORMS = ROOT / "data" / "figure_data" / "E_full_timeseries_storm_intervals.csv"
OUT = ROOT / "figures" / "E_full_timeseries"

df = pd.read_csv(IN_TS, parse_dates=["datetime"])
storms = pd.read_csv(IN_STORMS, parse_dates=["start", "end"])

fig, ax1 = plt.subplots(figsize=(13, 6))

e0 = df["E_J2_mean_Jkg"].iloc[0]
ax1.plot(df["datetime"], (df["E_J2_mean_Jkg"] - e0) / 1e3, "-", lw=0.9, color="#C44E52",
         label=r"$E_{J2}$, 3h bin mean (despiked)")
ax1.set_ylabel(r"$E_{J2} - E_{J2}(t_0)$ [kJ kg$^{-1}$]" + f"\n($t_0$={df['datetime'].iloc[0]:%Y-%m-%d})",
               color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax1.set_xlabel("UTC")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
a0 = df["a_equiv_mean_km"].iloc[0]
ax2.plot(df["datetime"], df["a_equiv_mean_km"] - a0, "-", lw=0.9, color="#4C72B0", alpha=0.7,
         label=r"equivalent $a$, 3h bin mean")
ax2.set_ylabel(r"$a_{\mathrm{equiv}} - a_{\mathrm{equiv}}(t_0)$ [km]", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")

for i, (_, s) in enumerate(storms.iterrows()):
    ax1.axvspan(s["start"], s["end"], color="gray", alpha=0.18, lw=0,
                label="Kp$\\geq$6 (3h bin)" if i == 0 else None)

# Gannon stormに注記
gannon = storms[(storms["start"] >= "2024-05-10") & (storms["start"] < "2024-05-13")]
if len(gannon):
    mid = gannon["start"].min() + (gannon["end"].max() - gannon["start"].min()) / 2
    ax1.annotate("Gannon storm\n(2024-05-10/11)", xy=(mid, ax1.get_ylim()[1] * 0.92),
                 ha="center", va="top", fontsize=8, color="0.3")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

ax1.set_title("Full-mission $J_2$-corrected specific energy: monotonic orbital decay, "
              "accelerating with rising solar activity\nNinjaSat, 2024-03-31 to 2024-11-30 "
              "(3h bin means, despiked)")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
fig.savefig(OUT.with_suffix(".pdf"))
print(f"wrote {OUT.with_suffix('.png')}")
print(f"wrote {OUT.with_suffix('.pdf')}")
