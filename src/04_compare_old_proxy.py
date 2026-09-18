"""新旧の軌道減衰proxyを共通の3h時間軸に揃えて比較する。

対応TODO: #1 (軌道減衰指標の再検討), #4 (Savitzky-Golayフィルタのアーティファクト検証
             — 本スクリプトも --window 引数で新旧proxyを同一条件で平滑化できる),
          #6 (不確かさの定量化 — 欠測を跨がない微分。03_dEdt_proxy.pyと同じ方針を
             新旧両方の3hビン系列に適用する)

入力:
  data/processed/specific_energy.csv
      02_specific_energy.py の出力。新proxy (E_J2 由来の da/dt) の元データ。
  data/legacy_v1/altitude_diff_monthly_ext/gps_tle_average_altitude_fitting_all_severe_{YYMM}_monthly_ext.csv
      旧解析 (MT_code) の Δh = TLE高度 − GPS高度 (2軌道 ≈ 3h ビン平均, km)。
      月ごとに前後約5日のパディング付き。旧解析はこの系列を Savitzky-Golay
      (deriv=1) で微分したものを drag proxy として使っていた
      (orbit_gps_tle_3.ipynb セル17, estimate_air_density.ipynb セル1・4)。
      パディング区間はファイル間でビン境界が僅かにずれる (別々の2軌道窓の
      切り方に依存) ため、各月ファイルからその月の「コア期間」のみを採用し
      (例: *_2404_*.csv → 2024-04-01〜2024-04-30 のみ)、月をまたいだ
      オーバーラップ由来の二重計上/不整合を避ける。

出力:
  data/figure_data/proxy_comparison.csv
      共通の3hグリッド上での新旧proxyの時系列 (内部結合)。
  data/figure_data/proxy_comparison_stats.csv
      --window ごとの相関係数 (Pearson/Spearman) を蓄積するテーブル。
      TODO #4 (SGフィルタ窓幅の感度分析) 用。plots/plot_proxy_comparison.py が
      再計算なしで注記に使う。

手法:
  1. 新: specific_energy.csv の E_J2_Jkg, a_equiv_m を3hビン平均 →
     (オプションでSavitzky-Golay平滑化) → 中心差分で dE/dt →
     da/dt = (2a^2/mu)*dE/dt に換算。 (03_dEdt_proxy.py と同じ手法。
     本スクリプトは新旧を同一条件で比較するため独立に再計算する。)
  2. 旧: 上記の月別コア期間データを結合した altitude_diff (km) を3hビン平均 →
     (オプションでSavitzky-Golay平滑化) → 中心差分で d(Δh)/dt [m/day]。
     符号: Δh = TLE - GPS。ドラッグが強まるとGPS実高度がSGP4(TLE)予測より
     早く下がる → Δhが増加 → d(Δh)/dt > 0 が「減衰が進んでいる」を意味する
     と期待される。一方、新proxy da/dt は減衰時に負。よって物理的には
     新旧proxyは逆符号の相関 (負の相関) を持つと予想される。
     中心差分は np.gradient をそのまま使わず、前後ビンがともに3h間隔で
     存在する点のみ計算する (03_dEdt_proxy.py と同じ欠測跨ぎ対策。
     隣接ビン間隔が6hを超える = 欠測を跨ぐ点はNaN。新旧どちらの系列にも
     同じ関数 bin_and_differentiate を適用する)。
  3. 新旧を共通の3hグリッドで内部結合し (NaNの点は相関計算から自然に
     除外される)、Pearson/Spearman相関係数を計算。

実行:
  .venv/Scripts/python src/04_compare_old_proxy.py               # 平滑化なし
  .venv/Scripts/python src/04_compare_old_proxy.py --window 13    # 感度分析用
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_NEW = ROOT / "data" / "processed" / "specific_energy.csv"
IN_OLD_DIR = ROOT / "data" / "legacy_v1" / "altitude_diff_monthly_ext"
OUT_FIG = ROOT / "data" / "figure_data" / "proxy_comparison.csv"
OUT_STATS = ROOT / "data" / "figure_data" / "proxy_comparison_stats.csv"

MU = 3.986004418e14  # GM [m^3/s^2]
BIN = "3h"
SECONDS_PER_DAY = 86400.0

# 各月ファイルのパディングを除いた「コア期間」 [start, end)
MONTH_CORE_RANGE = {
    "2404": ("2024-04-01", "2024-05-01"),
    "2405": ("2024-05-01", "2024-06-01"),
    "2406": ("2024-06-01", "2024-07-01"),
    "2407": ("2024-07-01", "2024-08-01"),
    "2408": ("2024-08-01", "2024-09-01"),
    "2409": ("2024-09-01", "2024-10-01"),
    "2410": ("2024-10-01", "2024-11-01"),
    "2411": ("2024-11-01", "2024-12-01"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--window", type=int, default=None,
        help="Savitzky-Golay平滑化のウィンドウ幅 [点数] (polyorder=1固定, 奇数のみ)。"
             "新旧proxy両方に同じ設定を適用する。未指定なら平滑化なし。",
    )
    return p.parse_args()


def gap_aware_center_diff(times, values, bin_td=pd.Timedelta(BIN)):
    """前後ビンがともに bin_td 間隔で存在する点のみ中心差分 (per second) を返す。

    03_dEdt_proxy.py の gap_aware_center_diff と同じロジック。np.gradient は
    不等間隔を機械的に重み付けしてしまい、欠測ギャップを跨ぐ点で見かけの
    スパイクを生む (docs/2026-07-02_spike_diagnosis.md 改修方針2) ため使わない。
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    dt_sec = bin_td.total_seconds()
    gaps = np.diff(times.values).astype("timedelta64[s]").astype(np.float64)
    dt_prev = np.full(n, np.nan)
    dt_next = np.full(n, np.nan)
    dt_prev[1:] = gaps
    dt_next[:-1] = gaps
    valid = np.isclose(dt_prev, dt_sec) & np.isclose(dt_next, dt_sec)
    idx = np.where(valid)[0]
    out[idx] = (values[idx + 1] - values[idx - 1]) / (2.0 * dt_sec)
    return out


def bin_and_differentiate(times, values, window):
    """3hビン平均済みの時系列 (times, values) から中心差分 (per day) を計算。

    欠測を跨ぐ点 (隣接ビン間隔が6hを超える) はNaNにする。
    """
    if window is not None:
        values = savgol_filter(values, window_length=window, polyorder=1)
    dv_dt_per_s = gap_aware_center_diff(times, values)
    return dv_dt_per_s * SECONDS_PER_DAY


def load_new_proxy(window):
    df = pd.read_csv(IN_NEW, parse_dates=["datetime"]).set_index("datetime")
    binned = df.resample(BIN).agg(
        E_J2_mean=("E_J2_Jkg", "mean"),
        a_mean=("a_equiv_m", "mean"),
        n_samples=("E_J2_Jkg", "count"),
    )
    binned = binned[binned["n_samples"] > 0].copy()

    t = binned.index
    e = binned["E_J2_mean"].to_numpy()
    a = binned["a_mean"].to_numpy()

    dEdt_Jkg_per_s_grid = bin_and_differentiate(t, e, window) / SECONDS_PER_DAY
    dadt_m_per_day = (2.0 * a**2 / MU) * dEdt_Jkg_per_s_grid * SECONDS_PER_DAY

    return pd.DataFrame({
        "datetime": t,
        "E_J2_mean": e,
        "dadt_new_m_per_day": dadt_m_per_day,
    })


def load_old_proxy(window):
    files = sorted(glob.glob(str(IN_OLD_DIR / "*_monthly_ext.csv")))
    frames = []
    for f in files:
        # ファイル名末尾の "_{YYMM}_monthly_ext.csv" から月コードを取り出す
        stem = Path(f).stem  # ..._2404_monthly_ext
        yymm = stem.split("_")[-3]
        assert yymm in MONTH_CORE_RANGE, f"unrecognised month code {yymm} in {f}"
        start, end = MONTH_CORE_RANGE[yymm]

        d = pd.read_csv(f)
        # timestampの書式が行によって混在 (末尾のパディング切れ目でtzオフセット無し
        # の行がある) するとpandasのparse_datesが文字列型のまま返すことがあるため、
        # 明示的にutc=Trueでto_datetimeする。
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        d = d.dropna(subset=["timestamp", "altitude_diff"])
        mask = (d["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (
            d["timestamp"] < pd.Timestamp(end, tz="UTC")
        )
        frames.append(d.loc[mask, ["timestamp", "altitude_diff"]])

    old = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    n_dup = old.duplicated(subset="timestamp").sum()
    if n_dup:
        print(f"warning: {n_dup} duplicate timestamps in old proxy, dropping")
        old = old.drop_duplicates(subset="timestamp", keep="first")

    old = old.set_index("timestamp")
    binned = old.resample(BIN).agg(
        altitude_diff_km=("altitude_diff", "mean"),
        n_samples=("altitude_diff", "count"),
    )
    binned = binned[binned["n_samples"] > 0].copy()

    t = binned.index
    diff_km = binned["altitude_diff_km"].to_numpy()
    ddiff_dt_km_per_day = bin_and_differentiate(t, diff_km, window)
    ddeltahdt_old_m_per_day = ddiff_dt_km_per_day * 1000.0

    return pd.DataFrame({
        "datetime": t,
        "altitude_diff_km": diff_km,
        "ddeltahdt_old_m_per_day": ddeltahdt_old_m_per_day,
    })


def main() -> None:
    args = parse_args()
    if args.window is not None and args.window % 2 == 0:
        raise ValueError(f"--window must be odd, got {args.window}")

    new_df = load_new_proxy(args.window)
    old_df = load_old_proxy(args.window)

    merged = pd.merge(new_df, old_df, on="datetime", how="inner")
    print(f"new proxy bins   : {len(new_df)}")
    print(f"old proxy bins   : {len(old_df)}")
    print(f"common (merged)  : {len(merged)}")
    print(f"common time range: {merged['datetime'].min()} .. {merged['datetime'].max()}")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FIG, index=False)
    print(f"wrote {OUT_FIG} ({len(merged)} rows)")

    # 欠測跨ぎでNaNになった点 (bin_and_differentiate) は相関計算から除外する
    # (出力CSVにはNaNのまま残す。統計量のみ有効点で計算)。
    valid = merged["dadt_new_m_per_day"].notna() & merged["ddeltahdt_old_m_per_day"].notna()
    n_nan = int((~valid).sum())
    if n_nan:
        print(f"dropping {n_nan} gap-adjacent NaN bin(s) before correlation")
    x = merged.loc[valid, "dadt_new_m_per_day"].to_numpy()
    y = merged.loc[valid, "ddeltahdt_old_m_per_day"].to_numpy()
    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    print("\n=== correlation: da/dt (new, E_J2-based) vs d(altitude_diff)/dt (old) ===")
    print(f"window          : {args.window}")
    print(f"Pearson  r = {r_pearson:+.4f}  (p = {p_pearson:.3g})")
    print(f"Spearman r = {r_spearman:+.4f}  (p = {p_spearman:.3g})")
    print("(negative correlation is the naive physical expectation: old proxy is "
          "TLE-GPS so it grows during decay, while new da/dt is negative during decay)")

    # TODO #4 感度分析用に window ごとの相関係数を蓄積して保存
    # (plots/plot_proxy_comparison.py が再計算なしで注記できるようにするため)
    stat_row = pd.DataFrame([{
        "window": args.window if args.window is not None else -1,
        "n": len(x),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
    }])
    if OUT_STATS.exists():
        prev = pd.read_csv(OUT_STATS)
        prev = prev[prev["window"] != stat_row["window"].iloc[0]]
        stats = pd.concat([prev, stat_row], ignore_index=True).sort_values("window")
    else:
        stats = stat_row
    stats.to_csv(OUT_STATS, index=False)
    print(f"wrote {OUT_STATS} ({len(stats)} window setting(s) recorded)")


if __name__ == "__main__":
    main()
