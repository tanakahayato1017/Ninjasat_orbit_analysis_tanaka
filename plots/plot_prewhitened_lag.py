"""プリホワイトニング(帯域通過)後のEUV遅延相関曲線 (自己相関バイアス除去後)。

対応TODO: #1, #4, #7

入力:
  data/figure_data/prewhitened_lag_correlation.csv  (11_prewhitened_lag.py の出力)

出力: figures/prewhitened_lag_curves.png / .pdf
  7波長のラグ相関曲線 (EUV帯=暖色、FUV帯=寒色)、各曲線のピーク位置にマーカー、
  y=0線。「全波長57-69hに収斂し、波長ごとの分離が見えない」ことが読み取れる形。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_prewhitened_lag.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CORR = ROOT / "data" / "figure_data" / "prewhitened_lag_correlation.csv"
OUT = ROOT / "figures" / "prewhitened_lag_curves"

# plot_euv_lag_scan.py と同じ配色 (EUV帯=暖色、FUV帯=寒色) を踏襲し、
# リポジトリ内の図一式で波長->色の対応を統一する。
WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
EUV_BAND = ["256", "284", "304"]
FUV_BAND = ["1175", "1216", "1335", "1405"]
EUV_COLORS = dict(zip(EUV_BAND, ["#7f0000", "#d62728", "#ff9896"]))
FUV_COLORS = dict(zip(FUV_BAND, ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]))
COLORS = {**EUV_COLORS, **FUV_COLORS}
WL_LABEL = {"256": "25.6 nm", "284": "28.4 nm", "304": "30.4 nm",
            "1175": "117.5 nm", "1216": "121.6 nm (Ly-a)",
            "1335": "133.5 nm", "1405": "140.5 nm"}
PEAK_MIN_LAG_H = 6

corr = pd.read_csv(IN_CORR)
corr["wl"] = corr["wavelength"].str.replace("irr_", "", regex=False)

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.axhline(0, color="gray", lw=0.7, zorder=0)

peak_lags = []
for wl in WAVELENGTHS:
    s = corr[corr["wl"] == wl].sort_values("lag_hours")
    ax.plot(s["lag_hours"], s["spearman_r"], "-", lw=1.3, color=COLORS[wl],
            alpha=0.85, label=f"{WL_LABEL[wl]}")
    s6 = s[s["lag_hours"] >= PEAK_MIN_LAG_H]
    pk = s6.loc[s6["spearman_r"].idxmax()]
    peak_lags.append(pk["lag_hours"])
    ax.plot(pk["lag_hours"], pk["spearman_r"], "o", ms=8, mfc=COLORS[wl],
            mec="black", mew=0.7, zorder=5)

lo, hi = min(peak_lags), max(peak_lags)
ax.axvspan(lo, hi, color="0.5", alpha=0.12, zorder=0)
ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92,
        f"peaks converge\n{lo:.0f}-{hi:.0f}h", fontsize=8, color="dimgray",
        ha="center")

ax.set_xlabel("lag $L$ [h]  (band-passed EUV($t-L$) vs band-passed proxy($t$))")
ax.set_ylabel("Spearman $r$")
ax.set_title(
    "Prewhitened (band-passed) lag correlation: 24h-smooth minus 5-day trend\n"
    "removes the shared low-frequency autocorrelation (27-day rotation, seasonal\n"
    "trend) that produced the flat plateau in the raw lag curves; all 7\n"
    "wavelengths converge to a common weak peak, no EUV/FUV separation survives",
    fontsize=9.5)
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim(0, 120)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
print(f"peak lags per wavelength: {sorted(zip(WAVELENGTHS, peak_lags))}")
