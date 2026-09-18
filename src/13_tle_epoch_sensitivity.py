r"""旧解析(Δh = TLE - GPS)のTLEエポック選択(日次/週次/月次)に対する感度分析。

対応TODO: #3 (単一TLE・1ヶ月伝播の妥当性検証、R2 Major #1・Minor #2)

背景:
  R2は「単一TLEを1ヶ月伝播させておりSGP4精度は時間とともに劣化する。TLEエポック選択
  (日次/週次/月次)への感度が未検証」と指摘した (Major#1, Minor#2)。改訂で導入した
  新proxy (dE/dt, 02_specific_energy.py) はGPSテレメトリのみから計算されTLEを一切
  使わないため、この懸念は新手法では構造的に消滅する。本スクリプトは旧手法
  (Δh = h_TLE - h_GPS) のTLE選択モードへの感度を定量化し、(a) 旧結果の誤差の
  位置づけ、(b) 新手法の優位性の根拠、を査読回答用に揃える。

高度定義 (docs/2026-07-02_old_altitude_definition.md で監査済みの旧解析の実際の定義。
"c'"・"(i)" 候補、RMS最良一致):
  GPS側 h_GPS(t) = |ECEF(lat(t), lon(t), gpsAltitudeMeters(t))| - 6378.137 km
                   (ジオイド未加算, 旧コードの実装どおり)
  TLE側 h_TLE(t) = |r_SGP4(t)| - 6378.137 km  (TEME座標のノルム, 緯度依存の
                   楕円体半径式は旧コード同様未使用)
  ECEF変換はsrc/12_old_altitude_definition_audit.pyのWGS84実装(pyproj不使用)を
  そのまま再利用する。

TLE選択モード3種 (全テレメトリ時刻 t に対し h_TLE(t) を生成):
  monthly (旧論文方式): 暦月ごとに、その月の15日00:00 UTCに最も近いエポックの
    TLE1本を選び、月全体をそのTLEで伝播する (src/12と同じ"月央"選定基準)。
  weekly: 月曜0時UTC基準の暦週ごとに、その週の月曜0時に対し直近過去
    (エポック<=月曜0時)で最も近いTLEを1本選び、週全体をそのTLEで伝播する。
  daily: 各時刻 t に対し、直近過去(エポック<=t)で最も近いTLEを都度選び伝播する
    (ninjasat_tle_per_dayはほぼ日次間引きなので実質的に日次更新)。

感度指標:
  (a) Δh(t) = h_TLE - h_GPS を3hビン平均。モード間の差の時系列
      (monthly-daily, weekly-daily) のstdと、月内でのΔh_monthly(t)の
      days-since-epochに対する線形drift傾き[m/day]の分布(月ごと)。
  (b) 各モードのΔh(t)(3hビン)にSavitzky-Golay(window=13, polyorder=1, 旧論文方式.
      src/04_compare_old_proxy.pyと同じgap-aware中心差分)をかけてd(Δh)/dtを求め、
      GOES EUV(irr_304, irr_1216)との全期間ラグ相関(Spearman, lag=0..120h,
      3h刻み)を計算する。ピークラグ(lag>=6hの範囲内でargmax、src/10と同じ
      自明相関除外基準)がTLE選択モードで何時間動くかが本タスクの主結果。
  (c) SGP4伝播誤差の成長: daily TLE基準のh_TLE(t)を"真値に最も近い"基準とし、
      monthly TLEのh_TLE(t)との差 (h_TLE_monthly - h_TLE_daily) を
      days-since-epoch(その月のTLEエポックからの経過日数, 符号付き)の関数として
      日次ビン平均し、|diff|のabs(days-since-epoch)に対する線形回帰の傾きを
      誤差成長率[m/day]とする。

入力:
  data/tle/ninjasat_tle_per_day.csv   (日次TLE, TLE_LINE1/TLE_LINE2/EPOCH)
  data/processed/gps_qc.csv           (01_prepare_gps.pyのQC済みテレメトリ)
  data/processed/euv_3h.csv           (09_prepare_euv.pyの3hグリッドEUV)

出力:
  data/figure_data/tle_epoch_sensitivity.csv
      columns: mode, wavelength, lag_hours, spearman_r, n
  data/figure_data/tle_epoch_peaks.csv
      columns: mode, wavelength, peak_lag_hours, peak_r, n
  data/figure_data/tle_epoch_drift.csv
      columns: days_since_epoch, median_diff_m, mean_diff_m, std_diff_m, n
      (monthly TLEのh_TLEとdaily TLE基準h_TLEとの差, 日次ビン平均。SGP4伝播誤差成長)
  data/figure_data/tle_epoch_modediff_stats.csv
      columns: quantity, value, unit, note
      (monthly-daily, weekly-daily の3hビンΔh差のstd、月次driftスロープの分布統計、
      SGP4誤差成長率など、報告用のスカラー統計一式)

実行: .venv/Scripts/python src/13_tle_epoch_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress, rankdata
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parents[1]
IN_TLE = ROOT / "data" / "tle" / "ninjasat_tle_per_day.csv"
IN_GPS = ROOT / "data" / "processed" / "gps_qc.csv"
IN_EUV = ROOT / "data" / "processed" / "euv_3h.csv"
OUT_LAG = ROOT / "data" / "figure_data" / "tle_epoch_sensitivity.csv"
OUT_PEAKS = ROOT / "data" / "figure_data" / "tle_epoch_peaks.csv"
OUT_DRIFT = ROOT / "data" / "figure_data" / "tle_epoch_drift.csv"
OUT_STATS = ROOT / "data" / "figure_data" / "tle_epoch_modediff_stats.csv"

# WGS84 (src/12_old_altitude_definition_audit.py と同じ実装・同じ定数)
A_WGS84 = 6378.137               # 長半径(赤道半径) [km]
F_WGS84 = 1.0 / 298.257223563    # 扁平率
E2 = F_WGS84 * (2.0 - F_WGS84)   # 第一離心率^2
R0 = 6378.137                    # 旧コードが引いていた固定"地球半径"(=赤道半径) [km]

PERIOD_START = pd.Timestamp("2024-04-01", tz="UTC")
PERIOD_END = pd.Timestamp("2024-12-01", tz="UTC")  # exclusive, euv_3h.csvと同じ"full"期間

WAVELENGTHS = ["304", "1216"]
LAGS_H = list(range(0, 121, 3))
PEAK_MIN_LAG_H = 6          # lag<6hの自明な同時相関を除外 (src/10_euv_lag_correlationと同じ基準)
SG_WINDOW = 13               # 旧論文方式 (src/04_compare_old_proxy.py の --window 13 相当)
SG_POLYORDER = 1
BIN = "3h"
SECONDS_PER_DAY = 86400.0
MIN_N_CORR = 10


# --------------------------------------------------------------------------
# ECEF変換 (src/12_old_altitude_definition_audit.py と同一実装)
# --------------------------------------------------------------------------
def geodetic_to_ecef_km(lat_deg, lon_deg, h_km):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h_km) * np.cos(lat) * np.cos(lon)
    y = (n + h_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h_km) * np.sin(lat)
    return x, y, z


# --------------------------------------------------------------------------
# SGP4伝播
# --------------------------------------------------------------------------
def propagate_sgp4_teme(tle_line1: str, tle_line2: str, times_utc: pd.DatetimeIndex) -> np.ndarray:
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd_total = times_utc.to_julian_date().to_numpy()
    jd = np.floor(jd_total)
    fr = jd_total - jd
    err, r, _v = sat.sgp4_array(jd, fr)
    r = np.asarray(r, dtype=float)
    err = np.asarray(err)
    if np.any(err != 0):
        print(f"  warning: SGP4 nonzero error for {int(np.sum(err != 0))} sample(s) (set NaN)")
    r[err != 0, :] = np.nan
    return r  # (N, 3) km, TEME frame


def h_tle_for_group(tle_row: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    r_teme = propagate_sgp4_teme(tle_row["TLE_LINE1"], tle_row["TLE_LINE2"], times)
    return np.linalg.norm(r_teme, axis=1) - R0


def propagate_grouped(times: pd.DatetimeIndex, tle_idx: np.ndarray, tle: pd.DataFrame) -> np.ndarray:
    """timesの各要素に割り当てられたtle_idx(tle dfへのインデックス, NaN可)に従い、
    同じTLEを使う時刻をまとめてSGP4伝播しh_TLE[km]を返す。"""
    out = np.full(len(times), np.nan)
    valid = ~np.isnan(tle_idx)
    for idx_val in np.unique(tle_idx[valid]):
        mask = valid & (tle_idx == idx_val)
        out[mask] = h_tle_for_group(tle.loc[int(idx_val)], pd.DatetimeIndex(times[mask]))
    return out


# --------------------------------------------------------------------------
# TLE選択モード
# --------------------------------------------------------------------------
def select_daily_idx(times: pd.Series, tle: pd.DataFrame) -> np.ndarray:
    """各時刻について、直近過去(エポック<=t)で最も近いTLEのインデックスを返す。"""
    left = pd.DataFrame({"datetime": times.to_numpy()})
    right = tle[["EPOCH"]].reset_index().rename(columns={"index": "tle_idx"})
    merged = pd.merge_asof(left, right, left_on="datetime", right_on="EPOCH", direction="backward")
    return merged["tle_idx"].to_numpy(dtype=float)


def select_weekly_idx(times: pd.Series, tle: pd.DataFrame) -> np.ndarray:
    """月曜0時UTC基準の暦週ごとに、その週の月曜0時に対し直近過去で最も近いTLEを
    1本選び、週内の全時刻に同じインデックスを割り当てる。"""
    week_start = times.dt.floor("D") - pd.to_timedelta(times.dt.dayofweek, unit="D")
    uniq_weeks = pd.DataFrame({"week_start": np.sort(week_start.unique())})
    right = tle[["EPOCH"]].reset_index().rename(columns={"index": "tle_idx"})
    merged_weeks = pd.merge_asof(uniq_weeks, right, left_on="week_start", right_on="EPOCH",
                                  direction="backward")
    week_to_idx = dict(zip(merged_weeks["week_start"], merged_weeks["tle_idx"]))
    return week_start.map(week_to_idx).to_numpy(dtype=float)


def select_monthly_idx(times: pd.Series, tle: pd.DataFrame):
    """暦月ごとに、その月の15日00:00 UTCに最も近い(前後どちらでもよい)エポックの
    TLEを1本選び、月内の全時刻に同じインデックスを割り当てる (旧論文方式)。
    月->選定インデックスの対応辞書も返す (drift解析で再利用する)。"""
    year_month = times.dt.to_period("M")
    month_to_idx = {}
    for m in sorted(year_month.unique()):
        target = pd.Timestamp(year=m.year, month=m.month, day=15, tz="UTC")
        idx = int((tle["EPOCH"] - target).abs().idxmin())
        month_to_idx[m] = idx
    return year_month.map(month_to_idx).to_numpy(dtype=float), month_to_idx


# --------------------------------------------------------------------------
# 3hビン平均・SG平滑化・gap-aware中心差分 (src/04_compare_old_proxy.pyと同じ方式)
# --------------------------------------------------------------------------
def bin_3h(times: pd.Series, values: np.ndarray) -> pd.Series:
    s = pd.Series(values, index=pd.DatetimeIndex(times))
    return s.resample(BIN).mean()


def gap_aware_center_diff(times: pd.DatetimeIndex, values: np.ndarray,
                           bin_td: pd.Timedelta = pd.Timedelta(BIN)) -> np.ndarray:
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


def smooth_and_differentiate_m_per_day(times: pd.DatetimeIndex, delta_h_km: np.ndarray) -> np.ndarray:
    """Δh(km, 3hグリッド, NaN可)にSG(window=13,polyorder=1)をかけ、gap-aware中心差分
    で d(Δh)/dt [m/day] を返す。NaNを含む区間はsavgol_filterがそのまま伝播できない
    ため、有限値の連続run単位でSGをかける (03/04と同じ設計思想: 欠測を跨いで
    フィルタしない)。"""
    n = len(delta_h_km)
    smoothed = np.full(n, np.nan)
    finite = np.isfinite(delta_h_km)
    # 連続する有限値runを見つけてrunごとにSGをかける
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return smoothed
    run_breaks = np.where(np.diff(idx) != 1)[0]
    run_starts = np.concatenate(([0], run_breaks + 1))
    run_ends = np.concatenate((run_breaks, [len(idx) - 1]))
    for rs, re in zip(run_starts, run_ends):
        seg_idx = idx[rs:re + 1]
        if len(seg_idx) >= SG_WINDOW:
            smoothed[seg_idx] = savgol_filter(delta_h_km[seg_idx], window_length=SG_WINDOW,
                                               polyorder=SG_POLYORDER)
        else:
            smoothed[seg_idx] = delta_h_km[seg_idx]  # runが短すぎる場合は平滑化なしで通す
    d_per_s = gap_aware_center_diff(times, smoothed)
    return d_per_s * SECONDS_PER_DAY * 1000.0  # km/s -> m/day


# --------------------------------------------------------------------------
# EUVラグ相関 (src/10_euv_lag_correlation.py と同じ整数グリッド参照方式)
# --------------------------------------------------------------------------
class EuvLookup:
    def __init__(self, euv_df: pd.DataFrame):
        self.start = euv_df["datetime"].iloc[0]
        self.n = len(euv_df)
        self.arrays = {wl: euv_df[f"irr_{wl}"].to_numpy(dtype=float) for wl in WAVELENGTHS}
        step = euv_df["datetime"].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
            f"euv_3h.csv is not a continuous 3h grid: {step}"

    def grid_pos(self, times: pd.Series) -> np.ndarray:
        offset_h = (times - self.start) / pd.Timedelta(hours=1)
        return np.rint(offset_h.to_numpy() / 3.0).astype(np.int64)

    def lag_matrix(self, wavelength: str, pos0: np.ndarray, lags_h: list) -> np.ndarray:
        arr = self.arrays[wavelength]
        lag_bins = np.asarray(lags_h, dtype=np.int64) // 3
        pos = pos0[:, None] - lag_bins[None, :]
        valid = (pos >= 0) & (pos < self.n)
        out = np.full(pos.shape, np.nan)
        out[valid] = arr[pos[valid]]
        return out


def fast_spearman(x: np.ndarray, y: np.ndarray, min_n: int = MIN_N_CORR):
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


def peak_from_curve(lags: np.ndarray, rs: np.ndarray):
    mask = lags >= PEAK_MIN_LAG_H
    sub_r = rs[mask]
    if np.all(np.isnan(sub_r)):
        return np.nan, np.nan
    idx = np.nanargmax(sub_r)
    return lags[mask][idx], sub_r[idx]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    # ---------------- load ----------------
    tle = pd.read_csv(IN_TLE)
    tle["EPOCH"] = pd.to_datetime(tle["EPOCH"], format="mixed").dt.tz_localize("UTC")
    tle = tle.sort_values("EPOCH").drop_duplicates("EPOCH").reset_index(drop=True)
    print(f"TLE table: {len(tle)} rows, EPOCH {tle['EPOCH'].min()} .. {tle['EPOCH'].max()}")

    gps = pd.read_csv(IN_GPS, parse_dates=["datetime"])
    gps = gps[(gps["datetime"] >= PERIOD_START) & (gps["datetime"] < PERIOD_END)].reset_index(drop=True)
    print(f"GPS samples in period [{PERIOD_START.date()}, {PERIOD_END.date()}): {len(gps)}")

    # h_GPS (c'定義, ジオイド未加算, 旧コード実装. docs/2026-07-02_old_altitude_definition.md)
    h_msl_km = gps["gpsAltitudeMeters"].to_numpy() / 1e3
    lat = gps["gpsLatitude"].to_numpy()
    lon = gps["gpsLongitude"].to_numpy()
    x, y, z = geodetic_to_ecef_km(lat, lon, h_msl_km)
    r_gps = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    h_gps = r_gps - R0

    times = gps["datetime"]
    times_idx = pd.DatetimeIndex(times)

    # ---------------- TLE選択 (3モード) ----------------
    daily_idx = select_daily_idx(times, tle)
    weekly_idx = select_weekly_idx(times, tle)
    monthly_idx, month_to_idx = select_monthly_idx(times, tle)

    for name, idx in [("daily", daily_idx), ("weekly", weekly_idx), ("monthly", monthly_idx)]:
        n_missing = int(np.isnan(idx).sum())
        if n_missing:
            print(f"  {name}: {n_missing} sample(s) with no assignable TLE (dropped -> NaN h_TLE)")

    # ---------------- SGP4伝播 (モードごと) ----------------
    print("\npropagating SGP4 per mode (this groups samples by shared TLE)...")
    h_tle_daily = propagate_grouped(times_idx, daily_idx, tle)
    h_tle_weekly = propagate_grouped(times_idx, weekly_idx, tle)
    h_tle_monthly = propagate_grouped(times_idx, monthly_idx, tle)
    print(f"  daily  : {len(np.unique(daily_idx[~np.isnan(daily_idx)]))} distinct TLEs used")
    print(f"  weekly : {len(np.unique(weekly_idx[~np.isnan(weekly_idx)]))} distinct TLEs used")
    print(f"  monthly: {len(np.unique(monthly_idx[~np.isnan(monthly_idx)]))} distinct TLEs used "
          f"({len(month_to_idx)} calendar months)")

    delta_h_km = {
        "daily": h_tle_daily - h_gps,
        "weekly": h_tle_weekly - h_gps,
        "monthly": h_tle_monthly - h_gps,
    }

    # ---------------- 3hビン平均 (共通グリッド) ----------------
    binned = {mode: bin_3h(times, dh) for mode, dh in delta_h_km.items()}
    full_grid = pd.date_range(PERIOD_START, PERIOD_END, freq=BIN, inclusive="left")
    for mode in binned:
        binned[mode] = binned[mode].reindex(full_grid)
    delta_df = pd.DataFrame(binned)
    delta_df.index.name = "datetime"
    print(f"\n3h-binned Delta h: {len(delta_df)} grid points "
          f"(non-NaN: daily={delta_df['daily'].notna().sum()}, "
          f"weekly={delta_df['weekly'].notna().sum()}, "
          f"monthly={delta_df['monthly'].notna().sum()})")

    # ================= (a) モード間の差の時系列と統計 =================
    diff_monthly_daily_m = (delta_df["monthly"] - delta_df["daily"]) * 1000.0
    diff_weekly_daily_m = (delta_df["weekly"] - delta_df["daily"]) * 1000.0
    std_monthly_daily = float(diff_monthly_daily_m.std())
    std_weekly_daily = float(diff_weekly_daily_m.std())
    print("\n=== (a) mode-diff of 3h-binned Delta h (vs daily reference) ===")
    print(f"  std(monthly - daily) = {std_monthly_daily:.2f} m  "
          f"(n={int(diff_monthly_daily_m.notna().sum())})")
    print(f"  std(weekly  - daily) = {std_weekly_daily:.2f} m  "
          f"(n={int(diff_weekly_daily_m.notna().sum())})")

    # 月内driftスロープ: 各暦月について、その月のΔh_monthly(t)[m] を
    # days-since-epoch[day] に対して線形回帰した傾き[m/day]
    drift_slopes = []
    for m, epoch_idx in month_to_idx.items():
        epoch = tle.loc[epoch_idx, "EPOCH"]
        mstart = pd.Timestamp(year=m.year, month=m.month, day=1, tz="UTC")
        mend = mstart + pd.DateOffset(months=1)
        sub = delta_df.loc[(delta_df.index >= mstart) & (delta_df.index < mend), "monthly"].dropna()
        if len(sub) < 5:
            continue
        days_since_epoch = (sub.index - epoch) / pd.Timedelta(days=1)
        slope, intercept, rval, pval, stderr = linregress(days_since_epoch, sub.to_numpy() * 1000.0)
        drift_slopes.append({"month": str(m), "epoch": epoch, "slope_m_per_day": slope,
                              "r": rval, "p": pval, "n": len(sub)})
    drift_df = pd.DataFrame(drift_slopes)
    print("\n=== (a) within-month drift slope of Delta h_monthly(t) vs days-since-epoch ===")
    print(drift_df.to_string(index=False))
    slope_vals = drift_df["slope_m_per_day"].to_numpy()
    print(f"  slope distribution [m/day]: median={np.median(slope_vals):.3f}, "
          f"min={slope_vals.min():.3f}, max={slope_vals.max():.3f}, "
          f"mean|slope|={np.mean(np.abs(slope_vals)):.3f}")

    # ================= (b) d(Delta h)/dt のEUVラグ相関 (モードごと) =================
    print(f"\n=== (b) d(Delta h)/dt (SG window={SG_WINDOW}, polyorder={SG_POLYORDER}) "
          "lag correlation vs EUV ===")
    euv_df = pd.read_csv(IN_EUV, parse_dates=["datetime"])
    euv_df["datetime"] = euv_df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    euv_df = euv_df.sort_values("datetime").reset_index(drop=True)
    euv = EuvLookup(euv_df)

    grid_naive = pd.DatetimeIndex(delta_df.index).tz_localize(None)

    dhdt_by_mode = {}
    for mode in ["daily", "weekly", "monthly"]:
        dhdt = smooth_and_differentiate_m_per_day(grid_naive, delta_df[mode].to_numpy())
        dhdt_by_mode[mode] = dhdt
        print(f"  {mode:8s}: d(Delta h)/dt valid bins = {int(np.isfinite(dhdt).sum())} / {len(dhdt)}")

    lag_rows = []
    for mode, dhdt in dhdt_by_mode.items():
        valid = np.isfinite(dhdt)
        sel_times = pd.Series(grid_naive[valid])
        sel_dhdt = dhdt[valid]
        pos0 = euv.grid_pos(sel_times)
        for wl in WAVELENGTHS:
            M = euv.lag_matrix(wl, pos0, LAGS_H)
            for li, L in enumerate(LAGS_H):
                r, n = fast_spearman(M[:, li], sel_dhdt)
                lag_rows.append({"mode": mode, "wavelength": wl, "lag_hours": L,
                                  "spearman_r": r, "n": n})
    lag_df = pd.DataFrame(lag_rows)
    OUT_LAG.parent.mkdir(parents=True, exist_ok=True)
    lag_df.to_csv(OUT_LAG, index=False)
    print(f"\nwrote {OUT_LAG} ({len(lag_df)} rows)")

    peak_rows = []
    for (mode, wl), g in lag_df.groupby(["mode", "wavelength"], sort=False):
        lags = g["lag_hours"].to_numpy()
        rs = g["spearman_r"].to_numpy()
        peak_lag, peak_r = peak_from_curve(lags, rs)
        n_at_peak = g.loc[g["lag_hours"] == peak_lag, "n"]
        n_at_peak = int(n_at_peak.iloc[0]) if len(n_at_peak) and not np.isnan(peak_lag) else 0
        peak_rows.append({"mode": mode, "wavelength": wl, "peak_lag_hours": peak_lag,
                           "peak_r": peak_r, "n": n_at_peak})
    peaks_df = pd.DataFrame(peak_rows).sort_values(["wavelength", "mode"])
    peaks_df.to_csv(OUT_PEAKS, index=False)
    print(f"wrote {OUT_PEAKS} ({len(peaks_df)} rows)")
    print("\n=== peak lag by mode & wavelength ===")
    print(peaks_df.to_string(index=False))

    peak_range_by_wl = {}
    for wl in WAVELENGTHS:
        sub = peaks_df[peaks_df["wavelength"] == wl]["peak_lag_hours"].dropna()
        if len(sub):
            peak_range_by_wl[wl] = (float(sub.min()), float(sub.max()), float(sub.max() - sub.min()))
            print(f"  {wl} nm: peak lag range across modes = "
                  f"[{sub.min():.0f}, {sub.max():.0f}] h  (spread {sub.max()-sub.min():.0f} h)")

    # ================= (c) SGP4伝播誤差の成長 (monthly vs daily reference) =================
    print("\n=== (c) SGP4 propagation error growth: h_TLE_monthly - h_TLE_daily "
          "vs days-since-epoch ===")
    epoch_per_sample = np.array([tle.loc[int(idx), "EPOCH"] if not np.isnan(idx) else pd.NaT
                                  for idx in monthly_idx])
    valid_c = ~np.isnan(monthly_idx) & ~np.isnan(daily_idx) & np.isfinite(h_tle_monthly) & \
              np.isfinite(h_tle_daily)
    days_since_epoch = np.full(len(times), np.nan)
    days_since_epoch[valid_c] = np.array([
        (t - e) / pd.Timedelta(days=1) for t, e in zip(times[valid_c], epoch_per_sample[valid_c])
    ])
    diff_m = (h_tle_monthly - h_tle_daily) * 1000.0  # m

    day_bin = np.full(len(times), np.nan)
    day_bin[valid_c] = np.floor(days_since_epoch[valid_c])
    drift_bins = pd.DataFrame({"days_since_epoch": day_bin[valid_c], "diff_m": diff_m[valid_c]})
    drift_summary = drift_bins.groupby("days_since_epoch")["diff_m"].agg(
        median_diff_m="median", mean_diff_m="mean", std_diff_m="std", n="count"
    ).reset_index().sort_values("days_since_epoch")
    drift_summary.to_csv(OUT_DRIFT, index=False)
    print(f"wrote {OUT_DRIFT} ({len(drift_summary)} rows, "
          f"days_since_epoch range [{drift_summary['days_since_epoch'].min():.0f}, "
          f"{drift_summary['days_since_epoch'].max():.0f}])")

    # 誤差成長率: |diff|の中央値 vs |days_since_epoch| の線形回帰 (原点通過を仮定しない)
    abs_days = drift_summary["days_since_epoch"].abs().to_numpy()
    abs_median = drift_summary["median_diff_m"].abs().to_numpy()
    fit_mask = (abs_days >= 1) & np.isfinite(abs_median)
    if fit_mask.sum() >= 3:
        growth_slope, growth_intercept, growth_r, growth_p, growth_se = linregress(
            abs_days[fit_mask], abs_median[fit_mask])
    else:
        growth_slope = growth_intercept = growth_r = growth_p = np.nan
    print(f"  |h_TLE_monthly - h_TLE_daily| growth rate (linear fit, |days_since_epoch|>=1): "
          f"{growth_slope:.3f} m/day  (intercept={growth_intercept:.2f} m, r={growth_r:.3f}, "
          f"p={growth_p:.3g}, n_bins={int(fit_mask.sum())})")

    # ================= 統計まとめCSV =================
    stat_rows = [
        {"quantity": "std_monthly_minus_daily_deltah_3h", "value": std_monthly_daily,
         "unit": "m", "note": "std of (Delta h_monthly - Delta h_daily), 3h bins, full period"},
        {"quantity": "std_weekly_minus_daily_deltah_3h", "value": std_weekly_daily,
         "unit": "m", "note": "std of (Delta h_weekly - Delta h_daily), 3h bins, full period"},
        {"quantity": "within_month_drift_slope_median", "value": float(np.median(slope_vals)),
         "unit": "m/day", "note": "median across months of linear slope of "
                                   "Delta h_monthly(t) vs days-since-epoch"},
        {"quantity": "within_month_drift_slope_min", "value": float(slope_vals.min()),
         "unit": "m/day", "note": "min across months"},
        {"quantity": "within_month_drift_slope_max", "value": float(slope_vals.max()),
         "unit": "m/day", "note": "max across months"},
        {"quantity": "sgp4_error_growth_rate", "value": float(growth_slope),
         "unit": "m/day", "note": "linear fit of |h_TLE_monthly-h_TLE_daily| median vs "
                                   "|days_since_epoch|, bins with |days_since_epoch|>=1"},
    ]
    for wl, (lo, hi, spread) in peak_range_by_wl.items():
        stat_rows.append({"quantity": f"peak_lag_range_{wl}nm_min", "value": lo,
                           "unit": "h", "note": f"min peak lag across TLE modes, {wl} nm"})
        stat_rows.append({"quantity": f"peak_lag_range_{wl}nm_max", "value": hi,
                           "unit": "h", "note": f"max peak lag across TLE modes, {wl} nm"})
        stat_rows.append({"quantity": f"peak_lag_spread_{wl}nm", "value": spread,
                           "unit": "h", "note": f"max-min peak lag across TLE modes, {wl} nm"})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT_STATS, index=False)
    print(f"\nwrote {OUT_STATS} ({len(stats_df)} rows)")

    print("\n=== summary ===")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
