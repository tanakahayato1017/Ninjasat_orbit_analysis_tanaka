"""地磁気活動の全期間概観 (Kp時系列+静穏マスク、月別擾乱割合)。

対応TODO: #8 (地磁気活動の定量的評価)

入力:
  data/figure_data/geomag_quiet_mask_3h.csv    (16_diagnostics_figdata.py,
    解析対象期間2024-04-01〜2024-11-30の3hグリッドKp + 直近48h max Kp + 静穏フラグ)
  data/figure_data/geomag_monthly_summary.csv  (16_diagnostics_figdata.py,
    月別Kp>=3/Kp>=4の3hビン割合)

出力: figures/geomag_overview.png / .pdf (2段構成)
  (a) 全期間のKp時系列 (3hビン) + 直近48h max Kp<4 の静穏区間を背景緑帯表示、
      Kp>=6の磁気嵐に日付注記。
  (b) 月別のKp>=3 / Kp>=4 インターバル割合の棒グラフ。

すべて図データCSVから描く (再計算しない)。
実行: .venv/Scripts/python plots/plot_geomag_overview.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_3H = ROOT / "data" / "figure_data" / "geomag_quiet_mask_3h.csv"
IN_MONTHLY = ROOT / "data" / "figure_data" / "geomag_monthly_summary.csv"
OUT = ROOT / "figures" / "geomag_overview"

BIN_TD = pd.Timedelta("3h")
STORM_KP_THRESHOLD = 6.0
STORM_LABEL_MIN_PEAK_KP = 7.0  # 混雑を避けるため、注記はこれ以上のイベントのみ

df = pd.read_csv(IN_3H, parse_dates=["datetime"])
monthly = pd.read_csv(IN_MONTHLY)
monthly["month_dt"] = pd.to_datetime(monthly["month"])


def quiet_intervals(d: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """連続する quiet_kp4=True の3hビンをまとめて [start, end) 区間にする。"""
    times = d.loc[d["quiet_kp4"], "datetime"].sort_values().to_numpy()
    if len(times) == 0:
        return []
    out = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > np.timedelta64(3, "h"):
            out.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
            start = t
        prev = t
    out.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
    return out


def storm_events(d: pd.DataFrame) -> list[dict]:
    """連続する Kp>=6 の3hビンをまとめ、各イベントの [start,end)・ピークKp・
    ピーク時刻を返す。"""
    hot = d[d["Kp"] >= STORM_KP_THRESHOLD].sort_values("datetime")
    if len(hot) == 0:
        return []
    events = []
    times = hot["datetime"].to_numpy()
    kps = hot["Kp"].to_numpy()
    start_i = 0
    for i in range(1, len(times) + 1):
        if i == len(times) or times[i] - times[i - 1] > np.timedelta64(3, "h"):
            seg_t = times[start_i:i]
            seg_k = kps[start_i:i]
            pk = int(np.argmax(seg_k))
            events.append({
                "start": pd.Timestamp(seg_t[0]),
                "end": pd.Timestamp(seg_t[-1]) + BIN_TD,
                "peak_kp": float(seg_k[pk]),
                "peak_time": pd.Timestamp(seg_t[pk]),
            })
            start_i = i
    return events


qintervals = quiet_intervals(df)
events = storm_events(df)
print(f"quiet (kp_max48h<4) intervals: {len(qintervals)}, "
      f"quiet fraction: {df['quiet_kp4'].mean():.1%}")
print(f"storm (Kp>=6) events: {len(events)}")
for e in events:
    print(f"  {e['start']} .. {e['end']}  peak Kp={e['peak_kp']:.2f} "
          f"@ {e['peak_time']}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8.5),
                          gridspec_kw={"height_ratios": [1.3, 1]})

# ---------------- (a) 全期間Kp時系列 + 静穏マスク + 磁気嵐注記 ----------------
ax = axes[0]
for s, e in qintervals:
    ax.axvspan(s, e, color="#2ca02c", alpha=0.12, lw=0)
ax.bar(df["datetime"], df["Kp"], width=BIN_TD, align="edge",
       color="#4C72B0", edgecolor="none", linewidth=0, alpha=0.8)
ax.axhline(STORM_KP_THRESHOLD, color="#C44E52", lw=0.8, ls="--", alpha=0.7,
           label=f"Kp={STORM_KP_THRESHOLD:.0f} (storm threshold)")
ax.axhline(4.0, color="0.4", lw=0.7, ls=":", alpha=0.7)

label_toggle = 0
for e in events:
    if e["peak_kp"] < STORM_LABEL_MIN_PEAK_KP:
        continue
    y_text = 8.6 if label_toggle % 2 == 0 else 7.6
    label_toggle += 1
    ax.annotate(f"{e['peak_time'].month}/{e['peak_time'].day}",
                xy=(e["peak_time"], e["peak_kp"]), xytext=(e["peak_time"], y_text),
                fontsize=7.5, ha="center", color="#7f0000",
                arrowprops=dict(arrowstyle="-", color="#7f0000", lw=0.6, alpha=0.7))

ax.set_ylim(0, 9.5)
ax.set_ylabel("Kp (3h bin)")
ax.set_title(
    "(a) Full-period Kp index (2024-04-01 to 2024-11-30)\n"
    "green bands = quiet windows (trailing 48h max Kp < 4, i.e. 'quiet_kp4'); "
    "red dashed = storm threshold Kp=6; dates = storm peak Kp>="
    f"{STORM_LABEL_MIN_PEAK_KP:.0f}", fontsize=9.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25, axis="y")

# ---------------- (b) 月別 Kp>=3 / Kp>=4 割合 ----------------
ax = axes[1]
x = np.arange(len(monthly))
w = 0.38
ax.bar(x - w / 2, monthly["frac_kp_ge3"] * 100, width=w, color="#DD8452",
       label="Kp >= 3 (3h bin fraction)")
ax.bar(x + w / 2, monthly["frac_kp_ge4"] * 100, width=w, color="#C44E52",
       label="Kp >= 4 (3h bin fraction)")
ax.set_xticks(x)
ax.set_xticklabels(monthly["month_dt"].dt.strftime("%Y-%m"))
ax.set_ylabel("fraction of 3h bins [%]")
ax.set_title("(b) Monthly disturbed-interval fraction "
             "(Sep-Oct most disturbed, Jul quietest)", fontsize=9.5)
ax.set_ylim(0, 40)

# 最擾乱・最静穏月を注記 (棒の上、タイトルと重ならない高さに)
imax = monthly["frac_kp_ge3"].idxmax()
imin = monthly["frac_kp_ge3"].idxmin()
ax.annotate(f"most disturbed\n{monthly.loc[imax,'frac_kp_ge3']*100:.0f}%",
            xy=(imax - w / 2, monthly.loc[imax, "frac_kp_ge3"] * 100),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=7.5, color="#7f0000")
ax.annotate(f"quietest\n{monthly.loc[imin,'frac_kp_ge3']*100:.0f}%",
            xy=(imin - w / 2, monthly.loc[imin, "frac_kp_ge3"] * 100),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=7.5, color="#1f4e79")

ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
