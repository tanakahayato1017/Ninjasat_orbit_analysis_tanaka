"""全期間(2024-04〜11)のGOES EUV(irr_304, irr_1216)と軌道減衰proxy(-dE/dt)の
(R3.1 2026-09-07: 凡例ラベルを列名irr_XXXから波長表記 "30.4 nm (He II)" 等に変更)
(R3.2 2026-09-07: Reviewer 3 Comment 3 に従い上段(a)を全7チャネル表示に変更。配色はOkabe-Ito)
時系列を並べて示す (代表例=2024年7月のみだった図を全期間に拡張)。

対応TODO: #10 (3ヶ月全期間の時系列図を追加, R2 Minor #3)

入力:
  data/processed/euv_3h.csv    (09_prepare_euv.py の出力, 連続3hグリッド)
  data/processed/dEdt_3h.csv   (03_dEdt_proxy.py の出力, 平滑化なし + se列)
  data/processed/geomag_3h.csv (05_geomagnetic_indices.py の出力)

処理・図データ (data/figure_data/full_timeseries_euv_proxy.csv に保存):
  1. EUV: irr_304, irr_1216 をそれぞれ自身の中央値 (NaN除く) で正規化する
     (irr_XXX_norm = irr_XXX / median(irr_XXX), 無次元・中央値=1)。
  2. proxy: dEdt_Jkg_per_day に 10_euv_lag_correlation.py と同じ modified
     z-score (Iglewicz & Hoaglin) |z_mod| > 8.0 の外れ値除去を適用してから
     -dE/dt (neg_dEdt_Jkg_per_day) を作る。24hビン (=8 x 3hビン) の中心移動
     平均 (min_periods=3) を表示系列とする。
  3. 不確かさ帯: se_dEdt_Jkg_per_day (03_dEdt_proxy.py の誤差伝播列) を同じ
     24hローリング窓で伝播する。窓内のビンが独立とみなし
       se_roll = sqrt( sum_i se_dEdt_i^2 ) / n_valid
     (n_valid = 窓内の非NaNビン数) を採用する。図には neg_dEdt_roll ± 1 se_roll
     の帯として表示する。
  4. 磁気嵐マーク: geomag_3h.csv の Kp>=6 の3hビンを両パネル共通で薄い縦帯
     として表示する (連続するビンはまとめて1つの帯にする)。
  5. 3系列 (euv_3h, dEdt_3h, geomag_3h) は共通の3hグリッド (00,03,06,...UTC)
     に位相が揃っているため、datetimeキーでの外部結合 (outer merge) だけで
     時間軸を合わせられる (線形補間はしない)。

出力:
  data/figure_data/full_timeseries_euv_proxy.csv
    列: datetime, irr_304, irr_304_norm, irr_1216, irr_1216_norm, kp,
        storm_kp6, dEdt_Jkg_per_day, neg_dEdt_Jkg_per_day,
        neg_dEdt_roll24h_Jkg_per_day, se_neg_dEdt_roll24h_Jkg_per_day,
        n_valid_roll24h
  figures/full_timeseries_euv_proxy.png / .pdf

実行:
  .venv/Scripts/python plots/plot_full_timeseries_euv.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_EUV = ROOT / "data" / "processed" / "euv_3h.csv"
IN_DEDT = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_GEOMAG = ROOT / "data" / "processed" / "geomag_3h.csv"
# R3.2: 全7チャネル (波長順)。凡例ラベルは波長表記
WLS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
WL_LABELS = {"256": "25.6 nm", "284": "28.4 nm", "304": "30.4 nm (He II)",
             "1175": "117.5 nm", "1216": r"121.6 nm (Lyman-$\alpha$)",
             "1335": "133.5 nm", "1405": "140.5 nm"}
OUT_FIGDATA = ROOT / "data" / "figure_data" / "full_timeseries_euv_proxy.csv"
OUT_FIG = ROOT / "figures" / "full_timeseries_euv_proxy"

OUTLIER_THRESHOLD = 8.0   # 10_euv_lag_correlation.py と同じMAD基準
ROLL_WINDOW = 8           # 24h = 8 x 3h
ROLL_MIN_PERIODS = 3      # plot_dEdt_timeseries.py と同じ最小点数
KP_STORM_THRESHOLD = 6.0
BIN_TD = pd.Timedelta("3h")
# 10_euv_lag_correlation.py の period_bounds("full") と同じ定義。
# geomag_3h.csv はKp48h遡り計算のため前後にパディングを持つが、EUV/proxy
# データが存在しない期間まで図に含めないよう、解析対象期間で切る。
PERIOD_START = pd.Timestamp("2024-04-01")
PERIOD_END = pd.Timestamp("2024-12-01")


def to_naive_utc(s: pd.Series) -> pd.Series:
    if s.dt.tz is not None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def remove_outliers_mad(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """10_euv_lag_correlation.py / 07_validate_dEdt.py と同じ外れ値除去。"""
    n_nan = int(df[col].isna().sum())
    if n_nan:
        df = df.dropna(subset=[col]).copy()
    x = df[col].to_numpy()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mod_z = 0.6745 * (x - med) / mad
    mask = np.abs(mod_z) > threshold
    print(f"MAD outlier removal on '{col}': removed {int(mask.sum())} point(s) "
          f"(threshold=|z_mod|>{threshold})")
    return df.loc[~mask].copy()


def build_figure_data() -> pd.DataFrame:
    euv = pd.read_csv(IN_EUV, parse_dates=["datetime"])
    euv["datetime"] = to_naive_utc(euv["datetime"])
    euv = euv[["datetime"] + [f"irr_{w}" for w in WLS]].copy()
    for col in [f"irr_{w}" for w in WLS]:
        med = np.nanmedian(euv[col].to_numpy())
        euv[f"{col}_norm"] = euv[col] / med
        print(f"  {col}: median={med:.6g} (normalization reference)")

    dedt = pd.read_csv(IN_DEDT, parse_dates=["datetime"])
    dedt["datetime"] = to_naive_utc(dedt["datetime"])
    dedt = dedt[["datetime", "dEdt_Jkg_per_day", "se_dEdt_Jkg_per_day"]].copy()
    dedt = remove_outliers_mad(dedt, "dEdt_Jkg_per_day", OUTLIER_THRESHOLD)

    geomag = pd.read_csv(IN_GEOMAG, parse_dates=["datetime"])
    geomag["datetime"] = to_naive_utc(geomag["datetime"])
    geomag = geomag[["datetime", "Kp"]].rename(columns={"Kp": "kp"})

    merged = pd.merge(euv, dedt, on="datetime", how="outer")
    merged = pd.merge(merged, geomag, on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # dEdt_3h.csv の2024-03-31パディング分・geomag_3h.csvのKp48h遡り用パディング分は
    # 解析対象期間 (10_euv_lag_correlation.py の period_bounds("full") と同じ) の外
    # なので図の対象からは除く。ローリング平均は除く前の全区間で計算済みなので
    # 端の値も正しい (窓が周期外のデータを含んでいても値自体は妥当)。
    merged = merged[(merged["datetime"] >= PERIOD_START) &
                     (merged["datetime"] < PERIOD_END)].reset_index(drop=True)

    # grid consistency check (all three inputs share the same 3h phase)
    step = merged["datetime"].diff().dropna().unique()
    assert set(pd.Series(step)) <= {BIN_TD}, \
        f"merged full-period grid is not a clean 3h grid: {sorted(set(step))[:5]}"

    merged["storm_kp6"] = merged["kp"] >= KP_STORM_THRESHOLD
    merged["neg_dEdt_Jkg_per_day"] = -merged["dEdt_Jkg_per_day"]

    roll = merged["neg_dEdt_Jkg_per_day"].rolling(
        ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS)
    merged["neg_dEdt_roll24h_Jkg_per_day"] = roll.mean()
    merged["n_valid_roll24h"] = roll.count()

    se_sq_roll = (merged["se_dEdt_Jkg_per_day"] ** 2).rolling(
        ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS)
    sum_se_sq = se_sq_roll.sum()
    n_valid_se = merged["se_dEdt_Jkg_per_day"].rolling(
        ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS).count()
    merged["se_neg_dEdt_roll24h_Jkg_per_day"] = np.sqrt(sum_se_sq) / n_valid_se

    cols = (["datetime"] + [f"irr_{w}" for w in WLS] + [f"irr_{w}_norm" for w in WLS]
            + ["kp", "storm_kp6", "dEdt_Jkg_per_day", "neg_dEdt_Jkg_per_day",
            "neg_dEdt_roll24h_Jkg_per_day", "se_neg_dEdt_roll24h_Jkg_per_day",
            "n_valid_roll24h"])
    return merged[cols]


def storm_intervals(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """連続する Kp>=6 の3hビンをまとめて [start, end) 区間のリストにする。"""
    storm_times = df.loc[df["storm_kp6"].fillna(False), "datetime"].sort_values().to_numpy()
    if len(storm_times) == 0:
        return []
    intervals = []
    start = storm_times[0]
    prev = storm_times[0]
    for t in storm_times[1:]:
        if t - prev > np.timedelta64(3, "h"):
            intervals.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
            start = t
        prev = t
    intervals.append((pd.Timestamp(start), pd.Timestamp(prev) + BIN_TD))
    return intervals


def main() -> None:
    print("=== building figure data ===")
    df = build_figure_data()
    OUT_FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FIGDATA, index=False)
    print(f"wrote {OUT_FIGDATA} ({len(df)} rows, "
          f"{df['datetime'].min()} .. {df['datetime'].max()})")

    intervals = storm_intervals(df)
    print(f"storm (Kp>=6) intervals: {len(intervals)}")

    FS_LABEL = 17
    FS_TICK = 14
    FS_LEGEND = 13
    FS_TITLE = 15
    FS_PANEL = 16

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True)

    # --- top: all seven EUV/FUV channels, median-normalized (R3.2) ---
    ax = axes[0]
    ax.axhline(1.0, color="gray", lw=0.6)
    # R3.2b: 識別性優先のカテゴリカル配色 (Okabe-Ito, 色覚多様性対応)。短波長=寒色系、長波長=暖色系
    wl_colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]
    for w, c in zip(WLS, wl_colors):
        ax.plot(df["datetime"], df[f"irr_{w}_norm"], "-", lw=0.8, color=c,
                alpha=0.9, label=WL_LABELS[w])
    # 凡例帯(上部2行)のためにy上限を持ち上げる
    norm_all = df[[f"irr_{w}_norm" for w in WLS]].to_numpy()
    lo, hi = np.nanmin(norm_all), np.nanmax(norm_all)
    ax.set_ylim(lo - 0.03, hi + 0.42 * (hi - lo))
    ax.set_ylabel("GOES-18 EUVS irradiance\n(normalized by median)", fontsize=FS_LABEL)
    ax.set_title("Full-period GOES EUV/FUV irradiance and NinjaSat orbital-decay proxy "
                  "(2024-04 to 2024-11)", fontsize=FS_TITLE)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.005, 0.96, "(a)", transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

    # --- bottom: 24h rolling (-dE/dt) with +-1 SE band ---
    ax = axes[1]
    ax.axhline(0, color="gray", lw=0.6)
    roll = df["neg_dEdt_roll24h_Jkg_per_day"] / 1e3
    se = df["se_neg_dEdt_roll24h_Jkg_per_day"] / 1e3
    ax.fill_between(df["datetime"], roll - se, roll + se, color="#7f0000",
                     alpha=0.25, lw=0, label=r"24-h rolling mean $\pm$ 1 SE")
    ax.plot(df["datetime"], roll, "-", lw=1.1, color="#7f0000",
            label=r"$-dE_{J2}/dt$, 24-h rolling mean")
    ax.set_ylabel(r"$-dE_{J2}/dt$ [kJ kg$^{-1}$ day$^{-1}$]"
                  "\n(24h rolling mean)", fontsize=FS_LABEL)
    ax.set_xlabel("UTC", fontsize=FS_LABEL)
    ax.legend(loc="upper right", fontsize=FS_LEGEND)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.005, 0.96, "(b)", transform=ax.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

    # --- storm shading, both panels ---
    for a in axes:
        for i, (s, e) in enumerate(intervals):
            a.axvspan(s, e, color="crimson", alpha=0.15, lw=0,
                      label=r"$K_p \geq 6$ (3-h intervals)" if (a is axes[0] and i == 0) else None)
    # rebuild top legend so it includes the seven EUV lines and the storm marker.
    # 4列x2行、列優先充填のため行方向読みが波長順になるようハンドルを並べ替える:
    #   row1: 25.6 | 28.4 | 30.4 | 117.5   row2: 121.6 | 133.5 | 140.5 | Kp
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    axes[0].legend(handles, labels, loc="upper center", fontsize=FS_LEGEND, ncol=4,
                   framealpha=0.95)

    fig.autofmt_xdate()
    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": (b)の2行ylabelの左端が図の保存範囲から見切れるのを防ぐ
    fig.savefig(f"{OUT_FIG}.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{OUT_FIG}.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {OUT_FIG}.pdf / .png")


if __name__ == "__main__":
    main()
