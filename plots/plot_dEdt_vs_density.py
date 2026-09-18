"""dE/dt proxy と TU Delft独立密度データ (Swarm A/B POD, GRACE-FO ACC) の比較図。

対応TODO: #5 (独立の熱圏密度データによるdE/dt proxyの比較検証)

入力:  data/figure_data/dEdt_vs_density.csv        (07_validate_dEdt.py の出力,
         日平均・衛星別・正規化済み)
       data/figure_data/dEdt_vs_density_stats.csv  (07_validate_dEdt.py の出力,
         衛星別・日平均/3h参考の相関係数)
出力:  figures/dEdt_vs_density.pdf / .png

レイアウト:
  上段: (−dE/dt) 正規化と Swarm A・Swarm B・GRACE-FO 密度正規化 (いずれも日平均、
        中央値=1に規格化) の時系列重ね描き。
        (R3改訂 2026-08-23: Reviewer 3の質問に対応し、従来Swarm Aは下段のみ
        だったのを上段にも追加。高度順の説明はキャプション側に記載。)
        Yamamoto & Sori (2026) の2024年10月イベント(10/8, 10/11, 10/18-20)
        および2024-05-11 Gannon地磁気嵐を縦線で注記する。
  下段: 3衛星 (Swarm_A, Swarm_B, GRACE-FO) それぞれの散布図
        ((−dE/dt) vs 密度, 日平均) をPearson/Spearman相関係数とともに表示。

レイアウト変更はこのファイルだけを編集して再実行すればよい (再計算不要)。
実行:  .venv/Scripts/python plots/plot_dEdt_vs_density.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "figure_data" / "dEdt_vs_density.csv"
IN_STATS = ROOT / "data" / "figure_data" / "dEdt_vs_density_stats.csv"
OUT = ROOT / "figures" / "dEdt_vs_density"

# 上段の時系列比較に使う3衛星 (R3.1 2026-09-07: 凡例をA→B→GRACE-FOの直感順に変更、
#  散布図パネル(b)-(d)の並びとも一致。従来は高度近接順 B > GRACE-FO > A)
TIMESERIES_SATS = ["Swarm_A", "Swarm_B", "GRACE-FO"]
TIMESERIES_COLORS = {"Swarm_B": "#1f77b4", "GRACE-FO": "#2ca02c", "Swarm_A": "#ff7f0e"}

# 下段散布図に出す3衛星の表示順・色
SCATTER_SATS = ["Swarm_A", "Swarm_B", "GRACE-FO"]
SCATTER_COLORS = {"Swarm_A": "#ff7f0e", "Swarm_B": "#1f77b4", "GRACE-FO": "#2ca02c"}
SCATTER_LABELS = {"Swarm_A": "Swarm A (~477 km)",
                   "Swarm_B": "Swarm B (~515 km)",
                   "GRACE-FO": "GRACE-FO (~488 km)"}

# 注記するイベント (docs/2026-07-02_tudelft_validation.md 参照)
EVENTS = [
    ("2024-05-11", "Gannon storm"),
    ("2024-10-08", "Y&S 10/8"),
    ("2024-10-11", "Y&S 10/11"),
    ("2024-10-18", "Y&S 10/18-20"),
]

FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 14
FS_TITLE = 15
FS_SUBTITLE = 14
FS_TEXT = 11
FS_PANEL = 16

df = pd.read_csv(IN, parse_dates=["date"])
stats = pd.read_csv(IN_STATS)
stats_daily = stats[stats["resolution"] == "daily"].set_index("satellite")

fig = plt.figure(figsize=(14, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], wspace=0.32, hspace=0.32)

# --- 上段: 正規化時系列の重ね描き ---
ax1 = fig.add_subplot(gs[0, :])
ax1.axhline(1.0, color="gray", lw=0.6)

sub = df[df["satellite"] == "Swarm_B"].sort_values("date")
l1, = ax1.plot(sub["date"], sub["neg_dEdt_norm"], "-", lw=1.0, color="#d62728", alpha=0.85,
                label=r"$(-dE/dt)$ normalized (daily mean, median=1)")

lines = [l1]
for sat in TIMESERIES_SATS:
    sub = df[df["satellite"] == sat].sort_values("date")
    ln, = ax1.plot(sub["date"], sub["rho_mean_norm"], "-", lw=1.0,
                    color=TIMESERIES_COLORS[sat], alpha=0.75,
                    label=f"{SCATTER_LABELS[sat]} density, normalized")
    lines.append(ln)

ax1.set_ylabel("normalized amplitude\n(median = 1)", fontsize=FS_LABEL)
ax1.set_title(r"$(-dE/dt)$ proxy vs independent thermosphere density "
              "(TU Delft portal), daily mean, 2024-04 to 2024-11",
              fontsize=FS_TITLE)
# R3: Swarm A追加で4項目になったため2列×上部中央(データの空き領域)に配置し、
#     時系列との重なりを回避 (lower rightでは静穏期の曲線を覆ってしまう)
# R3.1 2026-09-07: 凡例は列優先で埋まる(2列: 1列目=先頭2項, 2列目=後半2項)ため、
#   視覚上の行方向読み( (-dE/dt) -> Swarm A / Swarm B -> GRACE-FO )になるよう
#   ハンドルを [(-dE/dt), Swarm B, Swarm A, GRACE-FO] の順で渡す
legend_order = [lines[0], lines[2], lines[1], lines[3]]
# R3.1: パネルラベル(a)(左上)との重なりを避けるため、凡例をわずかに右へ
ax1.legend(handles=legend_order, loc="upper center", fontsize=FS_LEGEND, ncol=2,
           framealpha=0.95, bbox_to_anchor=(0.55, 1.0), bbox_transform=ax1.transAxes)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(alpha=0.3)
# R3: 凡例を上部に置くため上限を4.0に拡大 (データ最大~2.9、3.2-4.0を凡例帯に)
ax1.set_ylim(0, 4.0)
ax1.tick_params(labelsize=FS_TICK)
ax1.text(0.005, 0.96, "(a)", transform=ax1.transAxes, fontsize=FS_PANEL,
          fontweight="bold", va="top", ha="left")

for date_str, label in EVENTS:
    d = pd.Timestamp(date_str, tz="UTC")
    ax1.axvline(d, color="gray", lw=0.7, ls="--", alpha=0.7)
    ax1.text(d, 3.15, label, rotation=90, fontsize=FS_TEXT, color="dimgray",
              ha="right", va="top")

# --- 下段: 衛星ごとの散布図 ---
panel_labels_bottom = ["(b)", "(c)", "(d)"]
for i, sat in enumerate(SCATTER_SATS):
    ax = fig.add_subplot(gs[1, i])
    sub = df[df["satellite"] == sat]
    ax.scatter(sub["rho_mean"] * 1e12, sub["neg_dEdt_Jkg_per_day"], s=10, alpha=0.5,
               color=SCATTER_COLORS[sat])
    ax.set_xlabel(r"density $\rho$ [$10^{-12}$ kg m$^{-3}$]", fontsize=FS_LABEL)
    if i == 0:
        ax.set_ylabel(r"$(-dE/dt)$" + "\n[J kg$^{-1}$ day$^{-1}$]", fontsize=FS_LABEL)
    r = stats_daily.loc[sat]
    ax.set_title(f"{SCATTER_LABELS[sat]}\n"
                 f"Pearson r={r['pearson_r']:+.3f}\n"
                 f"Spearman ρ={r['spearman_r']:+.3f} (n={int(r['n'])})",
                 fontsize=FS_SUBTITLE - 1)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.96, panel_labels_bottom[i], transform=ax.transAxes,
             fontsize=FS_PANEL, fontweight="bold", va="top", ha="left")

fig.tight_layout(rect=(0, 0.025, 1, 1))

OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=160)
print(f"wrote {OUT}.pdf / .png")
