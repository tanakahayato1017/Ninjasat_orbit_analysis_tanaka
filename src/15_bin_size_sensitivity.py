"""ビン幅感度分析: E_J2のビン平均・中心差分dE/dtを 95min(1軌道)/190min(2軌道)/
3h/6h の4通りのビン幅で再構成し、(a) 独立密度データ(Swarm B, GRACE-FO)との
日平均Spearman相関、(b) irr_304とのラグ相関ピークが、ビン幅の選択に対して
安定かどうかを確認する。

対応TODO: #9 (3時間サンプリング間隔の選定理由説明, R2 Minor #3の2つ目)

背景:
  現行パイプラインは3hビン (03_dEdt_proxy.py) を採用しているが、査読者は
  「なぜ3時間か。NinjaSatの軌道周期(~95分)で十分ではないか。他の間隔は検討したか」
  と指摘している (R2)。本スクリプトはGNSS生サンプル (10分間隔) から
  95min(1軌道)・190min(2軌道)・3h(現行)・6hの4通りのビン幅でE_J2ビン平均→
  dE/dtを独立に再構成し、主要な結論 (独立密度データとの相関の符号・強さ、
  EUVラグ相関のピーク位置) がビン幅に対して安定かどうかを直接確認する。

入力:
  data/processed/specific_energy.csv       (02_specific_energy.py の出力,
    despike後の生GPSサンプル, ~10分間隔)
  data/processed/tudelft_density_daily.csv (06_tudelft_density.py の出力,
    satellite in {Swarm_A, Swarm_B, GRACE-FO} の日平均密度)
  data/external/goes_euv/{YYMM}_csv_combined.csv (09_prepare_euv.py がMT_codeから
    コピー済みの生ファイル。ここでは irr_304 のみ、ビン幅ごとに独立に再ビン
    平均する。新規コピーは行わない)

手法:
  1. ビン平均: specific_energy.csv の E_J2_Jkg を各ビン幅で pandas.resample
     (origin=2024-04-01T00:00, label="left") により平均する。データが皆無の
     ビンはビン系列から除外する (03_dEdt_proxy.py と同じ)。
  2. 中心差分 (欠測跨ぎ>2ビン幅は差分しない):
       dE/dt[i] = (E[i+1] - E[i-1]) / (t[i+1] - t[i-1])
     ただし前後どちらかの隣接ビンとの実時間ギャップが 2×ビン幅 を超える場合は
     NaN とする (03の「厳密に1ビン幅」という条件よりも緩い閾値。ビン幅が
     小さいほど欠測ビンに当たる確率が上がるため、単発の欠測ビンをまたいでも
     差分を計算できるようにする)。分母には実際の経過時間 (t[i+1]-t[i-1]) を
     使う (固定Δtを仮定しない、np.gradientのような機械的重み付けもしない)。
  3. (a) 独立密度相関: 外れ値除去なしで得た -dE/dt を暦日 (UTC) 平均し、
     tudelft_density_daily.csv の Swarm_B・GRACE-FO の日平均密度と日付キーで
     内部結合、Spearman相関を計算する (07_validate_dEdt.py と同じ定義)。
     具体的なMAD外れ値除去 (07/10と同じ|z_mod|>8基準) は 02_specific_energy.py
     のサンプルレベルdespikeが既に劣化GNSS fixを源流で除去しているため
     (docs/2026-07-02_pipeline_remediation.md 改修1)、本スクリプトでは
     追加のdE/dtレベル外れ値除去を行わない (ビン幅を振ったときの素の安定性を
     見るのが目的のため)。
  4. (b) EUVラグ相関: SG平滑化はTODO #4の検証と同じ方針で「物理的な時間窓を
     固定する」= 39h相当に最も近い奇数点数をビン幅ごとに換算して使う
     (下記 sg_window_points)。EUV (irr_304) はビン幅と同じ origin/freq で
     独立に再ビン平均し (09_prepare_euv.py と同じQC: irr_304_flag_flag==0
     のみ有効)、proxyと同じ整数グリッド参照でラグ相関 (Spearman) を計算する。
     proxy側のみSG平滑化 (10_euv_lag_correlation.py の現行=非対称手法と同じ)。
     ラグは 0 からビン幅刻みで 120h 程度まで走査し、lag>=6h の範囲でのピークを
     採用する (10と同じ定義)。地磁気条件によるフィルタはかけない (全期間, all)。
  5. SG窓の換算 (sg_window_points):
       raw = 39h / bin_width_hours
       round(raw) が奇数ならそれを採用、偶数なら raw までの距離が近い方の
       奇数 (n-1 or n+1) を採用する。
     3hビンでは raw=13.0 (現行10_euv_lag_correlation.py のSG13と一致する
     ことを確認済み)。

出力:
  data/figure_data/bin_size_sensitivity.csv
    長形式の結果表。列:
      bin_width, bin_width_minutes, metric, satellite, wavelength,
      sg_window_points, sg_window_hours_actual, n, r, p_value, peak_lag_hours
    metric = "density_corr_daily": satellite (Swarm_B/GRACE-FO) ごとの
      (-dE/dt)日平均 vs 密度日平均のSpearman (r, p_valueに値, peak_lag_hoursはNaN)
    metric = "euv_lag_peak": irr_304とのラグ相関ピーク (rにpeak_r,
      peak_lag_hoursにピークラグ, p_valueはNaN)

判定 (コンソールに出力):
  ビン幅間で (a) 密度相関の符号・強さ、(b) ピークラグが安定かどうかを表示する。

実行:
  .venv/Scripts/python src/15_bin_size_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[1]
IN_ENERGY = ROOT / "data" / "processed" / "specific_energy.csv"
IN_DENSITY_DAILY = ROOT / "data" / "processed" / "tudelft_density_daily.csv"
RAW_EUV_DIR = ROOT / "data" / "external" / "goes_euv"
OUT_CSV = ROOT / "data" / "figure_data" / "bin_size_sensitivity.csv"

YYMM_LIST = ["2404", "2405", "2406", "2407", "2408", "2409", "2410", "2411"]
SATELLITES = ["Swarm_B", "GRACE-FO"]
WAVELENGTH = "304"

ORIGIN = pd.Timestamp("2024-04-01")
GAP_FACTOR = 2.0                 # 欠測跨ぎ>2ビン幅は差分しない
TARGET_SG_HOURS = 39.0           # 39h相当のSG窓を各ビン幅で換算
LAG_MAX_H = 120.0
PEAK_MIN_LAG_H = 6.0
MIN_N = 10

BIN_WIDTHS = [
    ("95min", pd.Timedelta(minutes=95)),   # 1軌道 (~95分)
    ("190min", pd.Timedelta(minutes=190)),  # 2軌道
    ("3h", pd.Timedelta(hours=3)),          # 現行パイプライン
    ("6h", pd.Timedelta(hours=6)),
]


# ------------------------------------------------------------ proxy: E,dEdt --

def gap_aware_center_diff_variable(t: pd.DatetimeIndex, values: np.ndarray,
                                    bin_td: pd.Timedelta,
                                    gap_factor: float = GAP_FACTOR) -> np.ndarray:
    """前後ビンとの実時間ギャップがともに gap_factor*bin_td 以下の点のみ、
    実際の経過時間を分母にした中心差分 (per second) を返す。"""
    n = len(values)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    threshold_sec = gap_factor * bin_td.total_seconds()
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(np.float64)
    dt_prev = np.full(n, np.nan)
    dt_next = np.full(n, np.nan)
    dt_prev[1:] = gaps
    dt_next[:-1] = gaps
    valid = (dt_prev <= threshold_sec) & (dt_next <= threshold_sec)
    idx = np.where(valid)[0]
    total_dt_sec = dt_prev[idx] + dt_next[idx]
    out[idx] = (values[idx + 1] - values[idx - 1]) / total_dt_sec
    return out


def build_dEdt(energy: pd.DataFrame, bin_td: pd.Timedelta) -> pd.DataFrame:
    """E_J2をビン平均し、欠測跨ぎ対策つき中心差分dE/dtを計算する。"""
    binned = energy.set_index("datetime")["E_J2_Jkg"].resample(
        bin_td, origin=ORIGIN, label="left"
    ).agg(["mean", "count"])
    binned = binned[binned["count"] > 0].copy()
    t = binned.index
    e = binned["mean"].to_numpy()
    dEdt_per_s = gap_aware_center_diff_variable(t, e, bin_td)
    out = pd.DataFrame({
        "datetime": t,
        "E_J2_mean": e,
        "n_samples": binned["count"].to_numpy(),
        "dEdt_Jkg_per_day": dEdt_per_s * 86400.0,
    })
    return out.reset_index(drop=True)


def sg_window_points(bin_td: pd.Timedelta, target_hours: float = TARGET_SG_HOURS) -> int:
    bin_hours = bin_td.total_seconds() / 3600.0
    raw = target_hours / bin_hours
    n = int(round(raw))
    if n < 3:
        n = 3
    if n % 2 == 0:
        n = n - 1 if abs((n - 1) - raw) <= abs((n + 1) - raw) else n + 1
    return n


def sg_compact(values: np.ndarray, window: int) -> np.ndarray:
    """有効値(非NaN)のみをコンパクトに並べてSG平滑化し、元の位置に戻す
    (14_smoothing_symmetry.py と同じ近似)。"""
    out = np.full(values.shape, np.nan)
    mask = ~np.isnan(values)
    n_valid = int(mask.sum())
    if n_valid < window:
        out[mask] = values[mask]
        return out
    out[mask] = savgol_filter(values[mask], window_length=window, polyorder=1)
    return out


# ---------------------------------------------------------------- (a) 密度 --

def density_corr(dEdt_df: pd.DataFrame, density_daily: pd.DataFrame) -> list[dict]:
    df = dEdt_df.copy()
    df["neg_dEdt"] = -df["dEdt_Jkg_per_day"]
    df = df.dropna(subset=["neg_dEdt"])
    day = df["datetime"].dt.floor("1D")
    daily = df.groupby(day)["neg_dEdt"].mean().rename("neg_dEdt_daily")
    daily_n = df.groupby(day)["neg_dEdt"].count().rename("n_bins")
    daily_df = pd.concat([daily, daily_n], axis=1).reset_index().rename(columns={"datetime": "date"})

    rows = []
    for sat in SATELLITES:
        dens_sat = density_daily[density_daily["satellite"] == sat][["date", "rho_mean"]]
        merged = pd.merge(daily_df, dens_sat, on="date", how="inner")
        n = len(merged)
        if n < MIN_N:
            r, p = np.nan, np.nan
        else:
            r, p = spearmanr(merged["neg_dEdt_daily"], merged["rho_mean"])
        rows.append({"satellite": sat, "n": n, "r": r, "p_value": p})
    return rows


# ------------------------------------------------------------------ (b) EUV --

def load_raw_euv_304() -> pd.Series:
    frames = []
    for yymm in YYMM_LIST:
        f = RAW_EUV_DIR / f"{yymm}_csv_combined.csv"
        d = pd.read_csv(f, usecols=["Time", "irr_304", "irr_304_flag_flag"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df.sort_values("Time").drop_duplicates("Time").set_index("Time")
    bad = df["irr_304_flag_flag"] != 0
    df.loc[bad, "irr_304"] = np.nan
    s = df["irr_304"]
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s


def bin_euv_304(euv_1min: pd.Series, bin_td: pd.Timedelta) -> pd.DataFrame:
    binned = euv_1min.resample(bin_td, origin=ORIGIN, label="left").mean()
    out = binned.reset_index()
    out.columns = ["datetime", "irr_304"]
    return out


class GridLookup:
    """proxy/EUVのbin_td, originが揃っている前提で、整数グリッド参照する
    (10_euv_lag_correlation.py と同じ設計をビン幅一般化)。"""

    def __init__(self, arr: np.ndarray, start: pd.Timestamp, bin_td: pd.Timedelta):
        self.arr = arr
        self.start = start
        self.bin_hours = bin_td.total_seconds() / 3600.0
        self.n = len(arr)

    def grid_pos(self, times: pd.Series) -> np.ndarray:
        offset_h = (times - self.start) / pd.Timedelta(hours=1)
        return np.rint(offset_h.to_numpy() / self.bin_hours).astype(np.int64)

    def lag_matrix(self, pos0: np.ndarray, lag_bins: np.ndarray) -> np.ndarray:
        pos = pos0[:, None] - lag_bins[None, :]
        valid = (pos >= 0) & (pos < self.n)
        out = np.full(pos.shape, np.nan)
        out[valid] = self.arr[pos[valid]]
        return out


def fast_spearman(x: np.ndarray, y: np.ndarray, min_n: int = MIN_N):
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, n


def euv_lag_peak(dEdt_df: pd.DataFrame, euv_binned: pd.DataFrame,
                  bin_td: pd.Timedelta, window: int) -> dict:
    proxy_raw = (-dEdt_df["dEdt_Jkg_per_day"]).to_numpy()
    proxy_times = dEdt_df["datetime"]
    proxy_smoothed = sg_compact(proxy_raw, window)

    euv_start = euv_binned["datetime"].iloc[0]
    euv_arr = euv_binned["irr_304"].to_numpy(dtype=float)
    lookup = GridLookup(euv_arr, euv_start, bin_td)

    valid = ~np.isnan(proxy_smoothed)
    sel_times = proxy_times[valid]
    sel_proxy = proxy_smoothed[valid]
    pos0 = lookup.grid_pos(sel_times)

    bin_hours = bin_td.total_seconds() / 3600.0
    n_steps = int(round(LAG_MAX_H / bin_hours))
    lag_bins = np.arange(0, n_steps + 1, dtype=np.int64)
    lag_hours = lag_bins * bin_hours

    M = lookup.lag_matrix(pos0, lag_bins)
    rs = np.full(len(lag_bins), np.nan)
    ns = np.full(len(lag_bins), 0, dtype=int)
    for li in range(len(lag_bins)):
        r, n = fast_spearman(M[:, li], sel_proxy)
        rs[li] = r
        ns[li] = n

    mask = lag_hours >= PEAK_MIN_LAG_H
    sub_r = rs[mask]
    if np.all(np.isnan(sub_r)):
        return {"peak_lag_hours": np.nan, "peak_r": np.nan, "n": 0}
    idx = np.nanargmax(sub_r)
    peak_lag = lag_hours[mask][idx]
    peak_r = sub_r[idx]
    peak_n = ns[mask][idx]
    return {"peak_lag_hours": float(peak_lag), "peak_r": float(peak_r), "n": int(peak_n)}


# ---------------------------------------------------------------- main -----

def main() -> None:
    energy = pd.read_csv(IN_ENERGY, parse_dates=["datetime"])
    energy["datetime"] = energy["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    energy = energy.sort_values("datetime").reset_index(drop=True)

    density_daily = pd.read_csv(IN_DENSITY_DAILY, parse_dates=["datetime"])
    density_daily = density_daily.rename(columns={"datetime": "date"})
    density_daily["date"] = density_daily["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    print("loading raw GOES EUV 1-min irr_304 (from data/external/goes_euv, already copied by 09)...")
    euv_1min = load_raw_euv_304()
    print(f"  {len(euv_1min)} 1-min rows, {euv_1min.notna().sum()} valid (flag==0)")

    rows = []
    print(f"\n{'bin_width':>8s}  {'sg_pts':>6s}  {'sg_hrs':>7s}  n_bins  n_diff_valid")
    for label, bin_td in BIN_WIDTHS:
        dEdt_df = build_dEdt(energy, bin_td)
        n_valid_diff = dEdt_df["dEdt_Jkg_per_day"].notna().sum()
        window = sg_window_points(bin_td)
        sg_hours_actual = window * bin_td.total_seconds() / 3600.0
        print(f"{label:>8s}  {window:6d}  {sg_hours_actual:7.2f}  "
              f"{len(dEdt_df):6d}  {n_valid_diff:6d}")

        bin_minutes = bin_td.total_seconds() / 60.0

        # (a) 独立密度相関
        for r in density_corr(dEdt_df, density_daily):
            rows.append({
                "bin_width": label, "bin_width_minutes": bin_minutes,
                "metric": "density_corr_daily", "satellite": r["satellite"],
                "wavelength": np.nan, "sg_window_points": np.nan,
                "sg_window_hours_actual": np.nan, "n": r["n"], "r": r["r"],
                "p_value": r["p_value"], "peak_lag_hours": np.nan,
            })

        # (b) EUV(304) ラグ相関ピーク
        euv_binned = bin_euv_304(euv_1min, bin_td)
        peak = euv_lag_peak(dEdt_df, euv_binned, bin_td, window)
        rows.append({
            "bin_width": label, "bin_width_minutes": bin_minutes,
            "metric": "euv_lag_peak", "satellite": np.nan,
            "wavelength": WAVELENGTH, "sg_window_points": window,
            "sg_window_hours_actual": sg_hours_actual, "n": peak["n"],
            "r": peak["peak_r"], "p_value": np.nan,
            "peak_lag_hours": peak["peak_lag_hours"],
        })

    result = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(result)} rows)")

    print("\n=== (a) density correlation ((-dE/dt) daily mean vs density daily mean, Spearman) ===")
    for label, _ in BIN_WIDTHS:
        sub = result[(result["bin_width"] == label) & (result["metric"] == "density_corr_daily")]
        parts = [f"{r['satellite']}: rho={r['r']:+.3f} (n={int(r['n'])}, p={r['p_value']:.2e})"
                 for _, r in sub.iterrows()]
        print(f"  {label:>8s}  " + "   ".join(parts))

    print("\n=== (b) irr_304 lag correlation peak (lag>=6h, proxy SG=39h-equivalent) ===")
    for label, _ in BIN_WIDTHS:
        sub = result[(result["bin_width"] == label) & (result["metric"] == "euv_lag_peak")].iloc[0]
        print(f"  {label:>8s}  sg={int(sub['sg_window_points'])}pt"
              f"({sub['sg_window_hours_actual']:.1f}h)  "
              f"peak_lag={sub['peak_lag_hours']:.2f}h  peak_r={sub['r']:.3f}  n={int(sub['n'])}")


if __name__ == "__main__":
    main()
