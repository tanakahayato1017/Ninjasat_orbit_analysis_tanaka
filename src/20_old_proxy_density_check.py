"""旧proxy d(Δh)/dt（TLE-GPS高度差の日次差分）と独立熱圏密度データの整合性チェック。

対応TODO: #6 (不確かさの定量化・旧手法の位置づけ), R2 Major#1関連
  (旧proxyが「本物の密度シグナル」を捉えていたことを定量的に示し、改訂稿で
  旧手法を「整合性チェックとして保持」する材料とする)

背景:
  旧解析 (MT_code, orbit_gps_tle_3.ipynb) は Δh = TLE高度 - GPS高度 (2軌道 ≈ 3h
  ビン平均) を Savitzky-Golay 微分した d(Δh)/dt を drag proxy として使っていた。
  04_compare_old_proxy.py はこれをSG平滑化した新proxyと比較し弱い負相関を報告して
  いるが、旧proxy単体が独立密度データ (TU Delft) とどの程度整合するかは未検証
  だった。インライン計算で ρ=+0.62〜+0.70 程度の相関が判明しており、本スクリプトは
  それを再現可能な形式にする。

入力:
  data/figure_data/proxy_comparison.csv       (04_compare_old_proxy.py の出力,
      --window未指定=平滑化なしの実行結果。列 altitude_diff_km が旧proxyの
      3hビン平均 Δh [km] の元データ)
  data/processed/tudelft_density_daily.csv    (06_tudelft_density.py の出力。
      衛星別 (Swarm_A, Swarm_B, GRACE-FO) 日平均密度 rho_mean)

出力:
  data/figure_data/old_proxy_vs_density.csv        (日次ペア, 3衛星分)
  data/figure_data/old_proxy_vs_density_stats.csv  (衛星別 Pearson/Spearman)

手法:
  1. proxy_comparison.csv の altitude_diff_km (3hビン, datetime付き) を暦日
     (UTC, "1D") で単純平均する。1日あたりのビン数 (n_bins) が4未満の日
     (旧proxy側の欠測が多い日) は日平均の代表性が低いとして除外する
     (>=4/8ビンを採用基準とする)。
  2. 連続する暦日 (日付差がちょうど1日) のペアのみ、1日差分
       d(Δh)/dt [m/day] = (Δh_mean[day+1] - Δh_mean[day]) * 1000
     を計算する。日付が飛んでいる (欠測日を挟む) 箇所は差分を計算しない
     (03/04と同じ「欠測を跨がない」方針を日次版に適用)。
     符号: Δh = TLE - GPS。ドラッグが強まるとGPS実高度がSGP4予測より
     早く下がる -> Δhが増加 -> d(Δh)/dt > 0 が「減衰が進んでいる」に対応する
     (04_compare_old_proxy.py と同じ符号規約)。したがって密度が高いほど
     d(Δh)/dt が大きくなる -> **正相関が物理的期待**である
     (新proxyのように符号反転を挟まない: 新proxyは da/dt が減衰時に負なので
     -dE/dt を使うが、旧proxyのd(Δh)/dtはそのまま減衰時に正)。
  3. 衛星ごとの日平均密度 (tudelft_density_daily.csv) と日付で内部結合し、
     Pearson/Spearman相関係数を計算する。

実行:
  .venv/Scripts/python src/20_old_proxy_density_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_PROXY = ROOT / "data" / "figure_data" / "proxy_comparison.csv"
IN_DENSITY_DAILY = ROOT / "data" / "processed" / "tudelft_density_daily.csv"
OUT_PAIRS = ROOT / "data" / "figure_data" / "old_proxy_vs_density.csv"
OUT_STATS = ROOT / "data" / "figure_data" / "old_proxy_vs_density_stats.csv"

MIN_BINS_PER_DAY = 4  # 日平均の代表性を確保する最低3hビン数 (最大8/日)

# 参考: 新proxy (-dE/dt) vs 密度 の対応する相関値 (docs/2026-07-02_tudelft_validation.md,
# 改修後パイプライン, data/figure_data/dEdt_vs_density_stats.csv より)。本スクリプトの
# 出力とは独立に既知値として記録しておく (統計サマリの脚注・図の参考注記に使う)。
NEW_PROXY_SPEARMAN_REF = {
    "GRACE-FO": 0.906,
    "Swarm_A": 0.924,
    "Swarm_B": 0.852,
}


def daily_mean_altitude_diff(proxy: pd.DataFrame) -> pd.DataFrame:
    """altitude_diff_km を暦日平均する。MIN_BINS_PER_DAY未満の日は除外。"""
    s = proxy.set_index("datetime")["altitude_diff_km"].dropna()
    day = s.index.floor("D")
    grp = s.groupby(day)
    daily = grp.agg(altitude_diff_km_mean="mean", n_bins="count")
    daily = daily[daily["n_bins"] >= MIN_BINS_PER_DAY].copy()
    daily.index.name = "date"
    return daily.reset_index()


def one_day_diff_old_proxy(daily: pd.DataFrame) -> pd.DataFrame:
    """連続する暦日 (差がちょうど1日) のペアのみ1日差分 [m/day] を計算する。"""
    daily = daily.sort_values("date").reset_index(drop=True)
    dates = daily["date"].to_numpy()
    values_km = daily["altitude_diff_km_mean"].to_numpy()

    gap_days = np.full(len(daily), np.nan)
    gap_days[1:] = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(float)
    is_consecutive = np.isclose(gap_days, 1.0)

    old_proxy_m_per_day = np.full(len(daily), np.nan)
    diff_km = np.diff(values_km, prepend=np.nan)
    old_proxy_m_per_day[is_consecutive] = diff_km[is_consecutive] * 1000.0

    out = daily.copy()
    out["old_proxy_ddeltahdt_m_per_day"] = old_proxy_m_per_day
    return out


def main() -> None:
    proxy = pd.read_csv(IN_PROXY, parse_dates=["datetime"])
    density_daily = pd.read_csv(IN_DENSITY_DAILY, parse_dates=["datetime"])
    density_daily = density_daily.rename(columns={"datetime": "date"})

    daily_alt = daily_mean_altitude_diff(proxy)
    print(f"daily altitude_diff_km (>= {MIN_BINS_PER_DAY} bins/day): "
          f"{len(daily_alt)} / {proxy['datetime'].dt.floor('D').nunique()} days")

    old_daily = one_day_diff_old_proxy(daily_alt)
    n_valid = int(old_daily["old_proxy_ddeltahdt_m_per_day"].notna().sum())
    print(f"old proxy daily series (consecutive-day 1-day diff): "
          f"{n_valid} / {len(old_daily)} days have a valid d(altitude_diff)/dt")

    old_daily_valid = old_daily.dropna(subset=["old_proxy_ddeltahdt_m_per_day"]).copy()

    print("\n=== correlation: old proxy d(altitude_diff)/dt [m/day] vs density ===")
    print("(sign convention: altitude_diff = TLE - GPS grows during decay, so "
          "d(altitude_diff)/dt > 0 during decay -> positive correlation with "
          "density is the physical expectation)")

    pair_rows = []
    stat_rows = []
    satellites = sorted(density_daily["satellite"].unique())
    for sat in satellites:
        dens_sat = density_daily[density_daily["satellite"] == sat][
            ["date", "rho_mean", "alt_mean", "n_samples"]
        ]
        merged = pd.merge(old_daily_valid, dens_sat, on="date", how="inner")
        n = len(merged)

        x = merged["old_proxy_ddeltahdt_m_per_day"].to_numpy()
        y = merged["rho_mean"].to_numpy()
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)

        new_ref = NEW_PROXY_SPEARMAN_REF.get(sat, np.nan)
        print(f"  {sat:10s}  n={n:3d}  Pearson r={r_p:+.4f} (p={p_p:.3g})  "
              f"Spearman rho={r_s:+.4f} (p={p_s:.3g})  "
              f"[new proxy Spearman rho (ref) = {new_ref:+.3f}]")

        pair_rows.append(pd.DataFrame({
            "date": merged["date"],
            "satellite": sat,
            "old_proxy_ddeltahdt_m_per_day": merged["old_proxy_ddeltahdt_m_per_day"],
            "rho_mean": merged["rho_mean"],
            "alt_mean_km": merged["alt_mean"] / 1000.0,
            "n_samples_density": merged["n_samples"],
        }))
        stat_rows.append({
            "satellite": sat,
            "n": n,
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_r": r_s,
            "spearman_p": p_s,
            "new_proxy_spearman_r_ref": new_ref,
        })

    pairs = pd.concat(pair_rows, ignore_index=True).sort_values(["satellite", "date"])
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"\nwrote {OUT_PAIRS} ({len(pairs)} rows)")

    stats = pd.DataFrame(stat_rows)
    stats.to_csv(OUT_STATS, index=False)
    print(f"wrote {OUT_STATS} ({len(stats)} rows)")

    print("\n=== summary: old proxy d(altitude_diff)/dt vs density, by satellite ===")
    for _, r in stats.iterrows():
        print(f"  {r['satellite']:10s}  n={int(r['n']):3d}  "
              f"Pearson r={r['pearson_r']:+.3f}  Spearman rho={r['spearman_r']:+.3f}  "
              f"(new proxy ref: {r['new_proxy_spearman_r_ref']:+.3f})")


if __name__ == "__main__":
    main()
