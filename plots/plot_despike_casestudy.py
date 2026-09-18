"""despikeケーススタディ2パネル図 (図3: despike_casestudy)。

対応TODO: #6 (不確かさの定量化)

入力:  data/figure_data/despike_casestudy.csv (17_despike_casestudy_figdata.py の出力)
出力:  figures/despike_casestudy.png / .pdf

レイアウト:
  (a) 2024-10-25 00-09 UTC の生E_J2サンプル。rolling median (13サンプル中心移動
      中央値) と ±10x大域MAD のしきい値帯を重ね、除去された04:45:13のサンプルを
      赤で強調する。
  (b) 2024-05-10〜12 (Gannon storm) の生E_J2。除去サンプルはゼロ (磁気嵐の
      ビンスケール応答は誤除去されない設計であることの証拠)。

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行: .venv/Scripts/python plots/plot_despike_casestudy.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "despike_casestudy.csv"
OUT = ROOT / "figures" / "despike_casestudy"

df = pd.read_csv(IN, parse_dates=["datetime"])
df["removed"] = df["removed"].astype(bool)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# --- (a) 2024-10-25: 除去された劣化fixの例 ---
ax = axes[0]
sub = df[df["panel"] == "oct25"].sort_values("datetime")
e0 = sub["roll_median_Jkg"].median()  # 表示用オフセット (中央値を差し引く)
ax.fill_between(sub["datetime"], (sub["threshold_lo_Jkg"] - e0) / 1e3,
                 (sub["threshold_hi_Jkg"] - e0) / 1e3, color="#4C72B0", alpha=0.15,
                 label=r"rolling median $\pm$ 10$\times$global MAD (despike threshold)")
ax.plot(sub["datetime"], (sub["roll_median_Jkg"] - e0) / 1e3, "-", lw=1.0, color="#4C72B0",
        alpha=0.8, label="13-sample centered rolling median")
kept = sub[~sub["removed"]]
rem = sub[sub["removed"]]
ax.plot(kept["datetime"], (kept["E_J2_Jkg"] - e0) / 1e3, ".-", ms=5, lw=0.6, color="0.25",
        label="raw $E_{J2}$ (kept)")
ax.scatter(rem["datetime"], (rem["E_J2_Jkg"] - e0) / 1e3, s=90, color="crimson",
           zorder=5, marker="X", label="removed (despike)")
for _, r in rem.iterrows():
    ax.annotate(f"  {r['datetime'].strftime('%H:%M:%S')} UTC\n  dev="
                f"{(r['E_J2_Jkg']-r['roll_median_Jkg'])/1e3:+.1f} kJ/kg",
                xy=(r["datetime"], (r["E_J2_Jkg"] - e0) / 1e3), fontsize=8, color="crimson",
                va="center")
ax.set_ylabel(r"$E_{J2} - $ median(rolling) [kJ kg$^{-1}$]")
ax.set_title("(a) 2024-10-25 00:00-09:00 UTC: degraded-fix outlier removed by despike")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# symlog: 通常のノイズ(~1kJ/kg)とスパイク(137kJ/kg)を同じ軸で見せる
ax.set_yscale("symlog", linthresh=5)
ax.set_ylim(-10, 200)

# --- (b) Gannon storm: 除去なし (実信号保存の証拠) ---
ax = axes[1]
sub = df[df["panel"] == "gannon"].sort_values("datetime")
e0 = sub["roll_median_Jkg"].median()
ax.fill_between(sub["datetime"], (sub["threshold_lo_Jkg"] - e0) / 1e3,
                 (sub["threshold_hi_Jkg"] - e0) / 1e3, color="#4C72B0", alpha=0.15,
                 label=r"rolling median $\pm$ 10$\times$global MAD (despike threshold)")
ax.plot(sub["datetime"], (sub["roll_median_Jkg"] - e0) / 1e3, "-", lw=1.0, color="#4C72B0",
        alpha=0.8, label="13-sample centered rolling median")
ax.plot(sub["datetime"], (sub["E_J2_Jkg"] - e0) / 1e3, ".-", ms=3, lw=0.6, color="0.25",
        label="raw $E_{J2}$ (all kept, 0 removed)")
n_removed = int(sub["removed"].sum())
ax.set_ylabel(r"$E_{J2} - $ median(rolling) [kJ kg$^{-1}$]")
ax.set_xlabel("UTC")
ax.set_title(f"(b) 2024-05-10 to 2024-05-12 (Gannon storm, Kp up to 9): "
             f"{n_removed} samples removed -- storm signal preserved")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

fig.suptitle("Sample-level despike case studies (02_specific_energy.py, "
             "13-sample rolling median, 10x global MAD threshold)", fontsize=12, y=0.995)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, 0.96))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
fig.savefig(OUT.with_suffix(".pdf"))
print(f"wrote {OUT.with_suffix('.png')}")
print(f"wrote {OUT.with_suffix('.pdf')}")
