"""dE/dt proxy (03_dEdt_proxy.py) を TU Delftポータルの独立熱圏密度データ
(06_tudelft_density.py) と比較し、日平均ベースの相関を評価する。

対応TODO: #5 (独立の熱圏密度データによるdE/dt proxyの比較検証)

入力:
  data/processed/dEdt_3h.csv               (03_dEdt_proxy.py の出力, 平滑化なし)
  data/processed/tudelft_density_daily.csv (06_tudelft_density.py の出力)
      satellite in {"Swarm_A"(~477km, 参照用), "Swarm_B"(~515km, NinjaSat ~500-530kmに最近接),
                    "GRACE-FO"(~488km)}

出力:
  data/figure_data/dEdt_vs_density.csv
      日平均・衛星ごとの正規化済み時系列比較データ。
      列: date, satellite, rho_mean, rho_mean_norm, alt_mean_km,
          n_samples_density, dEdt_Jkg_per_day, n_bins_dEdt,
          neg_dEdt_Jkg_per_day, neg_dEdt_norm
      *_norm は各系列をその共通期間内の中央値で割った無次元量 (中央値=1)。
  data/figure_data/dEdt_vs_density_stats.csv
      衛星ごとの (-dE/dt) vs 密度 の日平均・3h相関係数テーブル
      (plots/plot_dEdt_vs_density.py が再計算なしで散布図に注記するため)。

手法:
  1. 外れ値除去 (日平均前, 3hデータに対して):
     dEdt_Jkg_per_day の modified z-score (Iglewicz & Hoaglin 1993)
       z_mod = 0.6745 * (x - median(x)) / MAD(x)
     を計算する。標準的な「疑わしい外れ値」の目安は |z_mod|>3.5 だが、
     この基準ではこの系列は年間を通じてゆるやかに裾の重い分布のため
     10点 (1784点中) が引っかかり、うち8点は |z_mod| < 5.3 程度で
     `docs/2026-07-02_dEdt_proxy.md` が指摘する「1点ノイズが大きい」
     通常の分布の裾にすぎない。一方 2024-10-25 00:00 UTC, 06:00 UTC の
     2点だけ |z_mod| ≈ 24.7, 25.7 と他から一桁以上突出しており、これが
     ドキュメントに記録された既知の異常 (E_J2_mean の2024-10-25 03:00 UTC
     ビンの局所異常値がその前後ビンの中心差分に伝播したもの) に対応する。
     したがって、この既知の突出した1イベントのみを標的として除去するため
     閾値 |z_mod| > OUTLIER_THRESHOLD (デフォルト8.0) を採用した。
     ※現行の主データセットでは 03_dEdt_proxy.py のサンプルレベルdespike
     (改修方針1) がこの異常を上流で除去するため、このスクリーンで除去される
     ビンは0個である。スクリーンは安全策 (将来のデータ更新・再処理時の
     ガード) として保持する。除去した実際の行・z値をログに出力する。
     なお 03_dEdt_proxy.py の欠測跨ぎ差分対策 (改修方針2) により
     dEdt_Jkg_per_day には少数のNaNビン (欠測隣接) が含まれるようになった。
     これらは中央値/MAD計算の前にあらかじめ除外する (n_samples>0の欠測ビンが
     もともと系列に含まれないのと同じ扱い)。
  2. 残った3hビンを暦日 (UTC, "1D") で単純平均し、日平均 dE/dt を得る
     (n_bins_dEdt列に採用ビン数を記録。通常8だが、外れ値除去や元データの
     欠測により7以下の日もある)。
  3. 衛星ごとの日平均密度 (tudelft_density_daily.csv) と日付で内部結合。
  4. 物理的期待: 密度が高い(抵抗が強い)ほど dE/dt はより負に大きくなる
     ("−dE/dt" が正に大きくなる) はずなので、-dE/dt と密度の間の
     Pearson/Spearman相関を計算する (これが正相関なら期待と整合)。
     3h生データでも参考として同じ相関を計算するが、3hはビンごとの
     サンプル数・軌道位相由来のノイズが非常に大きいため
     (`docs/2026-07-02_dEdt_proxy.md` 参照)、日平均を主たる結果とする。
  5. 正規化: 各系列 (density, -dE/dt) をそれぞれの衛星別共通期間内の
     中央値で割り、比較しやすい無次元系列にする (中央値=1)。

実行:
  .venv/Scripts/python src/07_validate_dEdt.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_DEDT = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_DENSITY_DAILY = ROOT / "data" / "processed" / "tudelft_density_daily.csv"
OUT_FIG = ROOT / "data" / "figure_data" / "dEdt_vs_density.csv"
OUT_STATS = ROOT / "data" / "figure_data" / "dEdt_vs_density_stats.csv"

OUTLIER_THRESHOLD = 8.0  # modified z-score cutoff, see docstring


def remove_outliers_mad(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    n_nan = int(df[col].isna().sum())
    if n_nan:
        print(f"  dropping {n_nan} NaN bin(s) in '{col}' (gap-adjacent, see docstring) "
              "before MAD outlier detection")
        df = df.dropna(subset=[col]).copy()

    x = df[col].to_numpy()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mod_z = 0.6745 * (x - med) / mad
    mask = np.abs(mod_z) > threshold
    removed = df.loc[mask, ["datetime", col]].copy()
    removed["mod_z"] = mod_z[mask]

    print(f"MAD outlier removal on '{col}' (3h series, n={len(df)}):")
    print(f"  median={med:.2f}, MAD={mad:.2f}, threshold=|z_mod|>{threshold}")
    if len(removed):
        print(f"  removed {len(removed)} point(s):")
        for _, r in removed.iterrows():
            print(f"    {r['datetime']}  {col}={r[col]:.2f}  z_mod={r['mod_z']:+.2f}")
    else:
        print("  removed 0 points")

    return df.loc[~mask].copy(), removed


def daily_mean_dEdt(df_clean: pd.DataFrame) -> pd.DataFrame:
    g = df_clean.set_index("datetime").resample("1D")
    out = g.agg(
        dEdt_Jkg_per_day=("dEdt_Jkg_per_day", "mean"),
        n_bins_dEdt=("dEdt_Jkg_per_day", "count"),
    )
    out = out[out["n_bins_dEdt"] > 0].reset_index().rename(columns={"datetime": "date"})
    return out


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    return {"n": len(x), "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s}


def main() -> None:
    dedt = pd.read_csv(IN_DEDT, parse_dates=["datetime"])
    density_daily = pd.read_csv(IN_DENSITY_DAILY, parse_dates=["datetime"])
    density_daily = density_daily.rename(columns={"datetime": "date"})

    dedt_clean, removed = remove_outliers_mad(dedt, "dEdt_Jkg_per_day", OUTLIER_THRESHOLD)

    daily_dedt = daily_mean_dEdt(dedt_clean)
    daily_dedt["neg_dEdt_Jkg_per_day"] = -daily_dedt["dEdt_Jkg_per_day"]

    print(f"\ndaily dE/dt: {len(daily_dedt)} days "
          f"({daily_dedt['date'].min().date()} .. {daily_dedt['date'].max().date()})")

    # 参考: 3h生データ (外れ値除去後) での相関も計算しておく
    dedt_clean = dedt_clean.copy()
    dedt_clean["neg_dEdt_Jkg_per_day"] = -dedt_clean["dEdt_Jkg_per_day"]
    dedt_clean["date3h"] = dedt_clean["datetime"]

    comparison_rows = []
    stats_rows = []

    satellites = sorted(density_daily["satellite"].unique())
    for sat in satellites:
        dens_sat = density_daily[density_daily["satellite"] == sat].copy()

        merged_daily = pd.merge(daily_dedt, dens_sat, on="date", how="inner")
        n_common = len(merged_daily)
        print(f"\n=== {sat}: daily common days = {n_common} "
              f"({merged_daily['date'].min().date()} .. {merged_daily['date'].max().date()}) ===")

        stat_daily = correlate(
            merged_daily["neg_dEdt_Jkg_per_day"].to_numpy(),
            merged_daily["rho_mean"].to_numpy(),
        )
        print(f"  daily: Pearson r={stat_daily['pearson_r']:+.4f} (p={stat_daily['pearson_p']:.3g})  "
              f"Spearman rho={stat_daily['spearman_r']:+.4f} (p={stat_daily['spearman_p']:.3g})")

        # 参考: 3h生データ (外れ値除去後のみ) での相関。
        # tudelft 3hデータとの厳密なビン整合は取らず、日付キーで参考程度に
        # 突き合わせる (密度の日代表値 vs dE/dtの3hビン)。
        dens_for_3h = dens_sat.rename(columns={"date": "day"})
        dedt_clean["day"] = dedt_clean["datetime"].dt.floor("1D")
        merged_3h = pd.merge(dedt_clean, dens_for_3h, on="day", how="inner")
        stat_3h = correlate(
            merged_3h["neg_dEdt_Jkg_per_day"].to_numpy(),
            merged_3h["rho_mean"].to_numpy(),
        )
        print(f"  3h (ref, n={stat_3h['n']}): Pearson r={stat_3h['pearson_r']:+.4f} "
              f"(p={stat_3h['pearson_p']:.3g})  Spearman rho={stat_3h['spearman_r']:+.4f} "
              f"(p={stat_3h['spearman_p']:.3g})")

        stats_rows.append({"satellite": sat, "resolution": "daily", **stat_daily})
        stats_rows.append({"satellite": sat, "resolution": "3h_ref", **stat_3h})

        # --- 正規化した日次比較データ (主要出力) ---
        rho_med = merged_daily["rho_mean"].median()
        neg_dedt_med = merged_daily["neg_dEdt_Jkg_per_day"].median()
        out = pd.DataFrame({
            "date": merged_daily["date"],
            "satellite": sat,
            "rho_mean": merged_daily["rho_mean"],
            "rho_mean_norm": merged_daily["rho_mean"] / rho_med,
            "alt_mean_km": merged_daily["alt_mean"] / 1000.0,
            "n_samples_density": merged_daily["n_samples"],
            "dEdt_Jkg_per_day": merged_daily["dEdt_Jkg_per_day"],
            "n_bins_dEdt": merged_daily["n_bins_dEdt"],
            "neg_dEdt_Jkg_per_day": merged_daily["neg_dEdt_Jkg_per_day"],
            "neg_dEdt_norm": merged_daily["neg_dEdt_Jkg_per_day"] / neg_dedt_med,
        })
        comparison_rows.append(out)

    comparison = pd.concat(comparison_rows, ignore_index=True).sort_values(["satellite", "date"])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_FIG, index=False)
    print(f"\nwrote {OUT_FIG} ({len(comparison)} rows)")

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT_STATS, index=False)
    print(f"wrote {OUT_STATS} ({len(stats)} rows)")

    print("\n=== summary: daily Pearson/Spearman ((-dE/dt) vs density), by satellite ===")
    for _, r in stats[stats["resolution"] == "daily"].iterrows():
        print(f"  {r['satellite']:10s}  n={int(r['n']):3d}  "
              f"Pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.2e})  "
              f"Spearman rho={r['spearman_r']:+.3f} (p={r['spearman_p']:.2e})")


if __name__ == "__main__":
    main()
