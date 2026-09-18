r"""補足図S1-S3 (旧手法の図, original submission相当) 用データの再構成。

対応TODO: #12 ("GNSS-defined altitude"の定義を明記, R1指摘), 査読対応の最終図整理

背景:
  original submission の Figure 2-1/2-2/2-3 (PowerPointからのエクスポート, スライド
  タイトル文字が残存) を補足図S1-S3として再生成する。PDFは `git show
  develop:figures/Figure{1,2,3}.pdf` で取得済み (figures/legacy_originals/ に
  プレビューPNGとして保存済み)。これらは:
    S1 = Figure2-1: 1日分 (今回は2024-04-29/30) のGPS/TLE高度の1周回正弦フィット例、
         フィット点 (2軌道ビンの平均高度D) を重ね描き。
    S2 = Figure2-2: 2024年7月のTLE伝播高度・GNSS高度・その差 (2軸)。
    S3 = Figure2-3: 2024年7月の高度差分時系列とその平滑化微分 (2軸)。

  S2/S3は `src/12_old_altitude_definition_audit.py` の監査で使われたのと同じ
  アーカイブ済み月別ファイル (`data/legacy_v1/altitude_diff_monthly_ext/
  gps_tle_average_altitude_fitting_all_severe_2407_monthly_ext.csv`) の
  gps_average_altitude / tle_average_altitude / altitude_diff 列をそのまま
  (コア期間 2024-07-01〜08-01 に切って) 使う。これはoriginal submissionが実際に
  使っていた値そのものなので "as in the original submission" のキャプションに
  忠実。S3の平滑化微分は README/docs (2026-07-02_dEdt_proxy.md, old_altitude_definition.md)
  に記録された旧解析の実装 (savgol_filter, window=13点≈39h, polyorder=1, deriv=1) を
  altitude_diff系列に適用して再現する。

  S1は月別アーカイブファイルにビン平均 (D) しか無く生カデンス(~600s)のデータが
  無いため、`src/12_old_altitude_definition_audit.py` で監査・確定した旧解析の
  実際の定義 (GPS候補(c') = |ECEF(lat,lon,gpsAltitudeMeters)| - 6378.137,
  TLE候補(i) = |r_SGP4(TEME)| - 6378.137) とビン推定量 (IQR外れ値除去[k=2] +
  固定1/rev正弦+オフセットフィットのオフセットD, 2軌道幅ビン, 生サンプル数<=8は
  スキップ) を、この対象期間 (2024-04-29〜30) 用に生データ・SGP4から独立に
  再実装する (12のロジックを忠実に再利用、対象期間のみ変更)。TLEは対象期間の
  中央 (2024-04-29 12:00 UTC) に最も近いものを選ぶ (12の"月中央"選定を
  日単位の一枚絵用に読み替えたもの)。

入力:
  data/gnss/ninjasat_gnss_2024-04_2024-11.csv                 NinjaSat生GNSSテレメトリ
  data/tle/ninjasat_tle_per_day.csv                  TLE (日次間引き)
  data/legacy_v1/altitude_diff_monthly_ext/
      gps_tle_average_altitude_fitting_all_severe_2407_monthly_ext.csv

出力:
  data/figure_data/legacy_S1_raw_2404.csv    生カデンス (2024-04-29〜05-01,
      ~600s間隔): datetime, h_gps_km, h_tle_km
  data/figure_data/legacy_S1_bins_2404.csv   2軌道ビン (同期間): bin_center,
      D_gps_km, D_tle_km, n_raw
  data/figure_data/legacy_S2S3_monthly_2407.csv  2024-07 コア期間の月次アーカイブ値
      (アーカイブそのまま): timestamp, gps_average_altitude, tle_average_altitude,
      altitude_diff, ddeltahdt_km_per_s (savgol window=13,polyorder=1,deriv=1)

実行: .venv/Scripts/python src/21_legacy_figures_S1S3.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter
from sgp4.api import Satrec

warnings.filterwarnings("ignore", category=OptimizeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_GPS = ROOT / "data" / "gnss" / "ninjasat_gnss_2024-04_2024-11.csv"
RAW_TLE = ROOT / "data" / "tle" / "ninjasat_tle_per_day.csv"
RAW_OLD_2407 = (ROOT / "data" / "legacy_v1" / "altitude_diff_monthly_ext"
                / "gps_tle_average_altitude_fitting_all_severe_2407_monthly_ext.csv")

OUT_S1_RAW = ROOT / "data" / "figure_data" / "legacy_S1_raw_2404.csv"
OUT_S1_BINS = ROOT / "data" / "figure_data" / "legacy_S1_bins_2404.csv"
OUT_S2S3 = ROOT / "data" / "figure_data" / "legacy_S2S3_monthly_2407.csv"

# WGS84 (src/12_old_altitude_definition_audit.py と同じ定数)
A_WGS84 = 6378.137
F_WGS84 = 1.0 / 298.257223563
E2 = F_WGS84 * (2.0 - F_WGS84)
R0_OLD = 6378.137

# S1対象期間
S1_WINDOW_TARGET = pd.Timestamp("2024-04-29 12:00", tz="UTC")   # TLE選定の基準時刻
S1_START = pd.Timestamp("2024-04-29 00:00", tz="UTC")
S1_END = pd.Timestamp("2024-05-01 00:00", tz="UTC")
S1_LOAD_PAD = pd.Timedelta(hours=6)   # ビンフィット端効果対策の読み込みパディング

IQR_K = 2.0
MIN_COUNT = 8

# S2/S3対象期間 (旧解析と同じ2024年7月、コア期間のみ)
CORE_START_2407 = pd.Timestamp("2024-07-01", tz="UTC")
CORE_END_2407 = pd.Timestamp("2024-08-01", tz="UTC")
SG_WINDOW = 13     # 旧解析: 39h相当 (orbit_gps_tle_3.ipynb セル17 / estimate_air_density.ipynb)
SG_POLYORDER = 1


# --------------------------------------------------------------------------
# GPS QC (src/01_prepare_gps.py, src/12_old_altitude_definition_audit.py と同じ基準)
# --------------------------------------------------------------------------
def load_and_qc_gps(t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
    cols = ["gpsUtcTime", "gpsFixQual", "gpsLatitude", "gpsLongitude",
            "gpsAltitudeMeters", "gpsGroundSpeed", "gpsGeoid"]
    header = pd.read_csv(RAW_GPS, nrows=0).columns
    df = pd.read_csv(RAW_GPS, usecols=[c for c in cols if c in header])

    df = df[df["gpsFixQual"] == 1]
    alt_km = df["gpsAltitudeMeters"] / 1e3
    df = df[(alt_km > 400) & (alt_km < 600)]
    if "gpsGroundSpeed" in df.columns:
        df = df[(df["gpsGroundSpeed"] > 26_000) & (df["gpsGroundSpeed"] < 29_000)]
    else:
        # gpsGroundSpeed is not in the public GNSS release (data/gnss/README.md).
        # These 11 fixes failed the 26,000-29,000 km/h window on the full telemetry;
        # dropping them by time stamp reproduces the same selection.
        speed_qc_failed = [1712171839, 1712745189, 1713864378, 1717232842, 1719044415, 1721834762, 1725560793, 1727035637, 1727173061, 1730274387, 1732211515]
        df = df[~df["gpsUtcTime"].isin(speed_qc_failed)]
        print("NOTE: gpsGroundSpeed unavailable; applied the recorded speed-QC "
              "exclusion list instead (11 fixes).")
    df = df.sort_values("gpsUtcTime").drop_duplicates("gpsUtcTime", keep="first")

    df["datetime"] = pd.to_datetime(df["gpsUtcTime"], unit="s", utc=True)
    df = df[(df["datetime"] >= t_min) & (df["datetime"] <= t_max)].reset_index(drop=True)
    return df


def geodetic_to_ecef_km(lat_deg, lon_deg, h_km):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h_km) * np.cos(lat) * np.cos(lon)
    y = (n + h_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h_km) * np.sin(lat)
    return x, y, z


def select_nearest_tle(target: pd.Timestamp):
    tle = pd.read_csv(RAW_TLE)
    tle["EPOCH"] = pd.to_datetime(tle["EPOCH"], format="mixed")
    tle_epoch_utc = tle["EPOCH"].dt.tz_localize("UTC")
    idx = (tle_epoch_utc - target).abs().idxmin()
    row = tle.loc[idx]
    print(f"selected TLE: EPOCH={row['EPOCH']} (target {target}), "
          f"MEAN_MOTION={row['MEAN_MOTION']} rev/day")
    return row["TLE_LINE1"], row["TLE_LINE2"], float(row["MEAN_MOTION"])


def propagate_sgp4_teme(tle_line1: str, tle_line2: str, times_utc: pd.DatetimeIndex):
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd_total = times_utc.to_julian_date().to_numpy()
    jd = np.floor(jd_total)
    fr = jd_total - jd
    err, r, _v = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        n_bad = int(np.sum(err != 0))
        print(f"warning: SGP4 returned nonzero error code for {n_bad} samples (kept as NaN)")
    r = np.asarray(r, dtype=float)
    r[err != 0, :] = np.nan
    return r


def iqr_keep_mask(y: np.ndarray, k: float = IQR_K) -> np.ndarray:
    finite = np.isfinite(y)
    if finite.sum() < 4:
        return finite
    q1, q3 = np.nanpercentile(y[finite], [25, 75])
    iqr = q3 - q1
    return finite & (y >= q1 - k * iqr) & (y <= q3 + k * iqr)


def fit_offset(x: np.ndarray, y: np.ndarray, B: float) -> float:
    """固定周期(B=2pi/T)の正弦+オフセットをfitし、オフセットDを返す(失敗時はnan)。"""
    if len(x) < 5:
        return np.nan
    amp0 = (np.nanmax(y) - np.nanmin(y)) / 2.0
    mean0 = np.nanmean(y)
    try:
        def f(xx, A, C, D):
            return A * np.sin(B * xx + C) + D
        popt, _ = curve_fit(f, x, y, p0=[amp0, np.pi, mean0], maxfev=5000)
        return popt[2]
    except Exception:
        return np.nan


def bin_indices(elapsed_min: np.ndarray, orbital_period_min: float, min_count: int = MIN_COUNT):
    n_bins = int(np.ceil(elapsed_min.max() / orbital_period_min)) if len(elapsed_min) else 0
    out = []
    for i in range(n_bins):
        lo, hi = i * orbital_period_min, (i + 1) * orbital_period_min
        idx = np.where((elapsed_min >= lo) & (elapsed_min < hi))[0]
        out.append(idx if len(idx) > min_count else None)
    return out


# --------------------------------------------------------------------------
# S1: 生カデンス + 2軌道ビンフィット (2024-04-29〜30)
# --------------------------------------------------------------------------
def build_s1() -> None:
    print("=== S1: raw-cadence GPS/TLE altitude + bin fit (2024-04-29/30) ===")
    gps = load_and_qc_gps(S1_START - S1_LOAD_PAD, S1_END + S1_LOAD_PAD)
    print(f"QC'd raw GPS samples in padded window: {len(gps)}")

    tle1, tle2, mean_motion = select_nearest_tle(S1_WINDOW_TARGET)
    orbit_period_min = 1440.0 / mean_motion
    orbital_period = 2.0 * orbit_period_min   # 2軌道ビン幅 (src/12と同じ)
    B_fixed = 4.0 * np.pi / orbital_period
    print(f"single-orbit period T = {orbit_period_min:.3f} min, "
          f"2-orbit bin width = {orbital_period:.3f} min")

    # GPS候補(c'): |ECEF(lat,lon,gpsAltitudeMeters)| - R0  (旧解析の実際の実装)
    h_msl = gps["gpsAltitudeMeters"].to_numpy() / 1e3
    lat = gps["gpsLatitude"].to_numpy()
    lon = gps["gpsLongitude"].to_numpy()
    x_raw, y_raw, z_raw = geodetic_to_ecef_km(lat, lon, h_msl)
    h_gps = np.sqrt(x_raw ** 2 + y_raw ** 2 + z_raw ** 2) - R0_OLD

    # TLE候補(i): |r_SGP4(TEME)| - R0
    times = pd.DatetimeIndex(gps["datetime"])
    r_teme = propagate_sgp4_teme(tle1, tle2, times)
    h_tle = np.linalg.norm(r_teme, axis=1) - R0_OLD

    # ---- ビンフィット (padding込みの全区間で elapsed_min を計算, t0=読み込み窓の先頭) ----
    elapsed_min = (times - times.min()).total_seconds().to_numpy() / 60.0
    idx_list = bin_indices(elapsed_min, orbital_period)
    print(f"n candidate bins (padded window): {len(idx_list)}")

    bin_rows = []
    for idx in idx_list:
        if idx is None:
            continue
        x = elapsed_min[idx]
        center_time = times[idx].mean()
        y_gps = h_gps[idx]
        y_tle = h_tle[idx]
        keep_gps = iqr_keep_mask(y_gps)
        keep_tle = iqr_keep_mask(y_tle)
        d_gps = fit_offset(x[keep_gps], y_gps[keep_gps], B_fixed)
        d_tle = fit_offset(x[keep_tle], y_tle[keep_tle], B_fixed)
        bin_rows.append({"bin_center": center_time, "D_gps_km": d_gps,
                          "D_tle_km": d_tle, "n_raw": len(idx)})

    bins_df = pd.DataFrame(bin_rows).dropna(subset=["D_gps_km", "D_tle_km"])
    bins_df = bins_df.sort_values("bin_center").reset_index(drop=True)
    print(f"fitted bins (padded window, valid D): {len(bins_df)}")

    # ---- 出力: 表示対象期間 (S1_START..S1_END) のみに切って保存 ----
    raw_df = pd.DataFrame({"datetime": times, "h_gps_km": h_gps, "h_tle_km": h_tle})
    raw_out = raw_df[(raw_df["datetime"] >= S1_START) & (raw_df["datetime"] < S1_END)]
    bins_out = bins_df[(bins_df["bin_center"] >= S1_START) & (bins_df["bin_center"] < S1_END)]

    OUT_S1_RAW.parent.mkdir(parents=True, exist_ok=True)
    raw_out.to_csv(OUT_S1_RAW, index=False)
    bins_out.to_csv(OUT_S1_BINS, index=False)
    print(f"wrote {OUT_S1_RAW} ({len(raw_out)} rows)")
    print(f"wrote {OUT_S1_BINS} ({len(bins_out)} rows)")


# --------------------------------------------------------------------------
# S2/S3: 2024-07 アーカイブ済み月次ファイル (コア期間) + 平滑化微分
# --------------------------------------------------------------------------
def build_s2s3() -> None:
    print("\n=== S2/S3: archived 2024-07 monthly bins + smoothed derivative ===")
    old = pd.read_csv(RAW_OLD_2407)
    old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
    old = old.dropna(subset=["timestamp", "gps_average_altitude",
                              "tle_average_altitude", "altitude_diff"])
    old = old.sort_values("timestamp").reset_index(drop=True)

    core = old[(old["timestamp"] >= CORE_START_2407) &
               (old["timestamp"] < CORE_END_2407)].reset_index(drop=True)
    print(f"core-period (2024-07) archived bins: {len(core)}")

    # 旧解析の平滑化微分: savgol_filter(window=13,polyorder=1,deriv=1) を
    # altitude_diff [km] に適用し、d(Delta h)/dt を [km/s] で得る
    # (docs/2026-07-02_dEdt_proxy.md, old_altitude_definition.md 参照;
    #  savgol_filter は等間隔仮定のため、ビン間隔の中央値を delta として使う
    #  (~189分=2軌道幅、旧解析の実装と同じ近似))
    dt_seconds = core["timestamp"].diff().dt.total_seconds().median()
    print(f"median bin spacing: {dt_seconds:.1f} s ({dt_seconds/60:.2f} min)")

    values = core["altitude_diff"].to_numpy()
    if len(values) >= SG_WINDOW:
        deriv = savgol_filter(values, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
                               deriv=1, delta=dt_seconds)
    else:
        deriv = np.full_like(values, np.nan)
    core["ddeltahdt_km_per_s"] = deriv

    OUT_S2S3.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(OUT_S2S3, index=False)
    print(f"wrote {OUT_S2S3} ({len(core)} rows, "
          f"{core['timestamp'].min()} .. {core['timestamp'].max()})")


def main() -> None:
    build_s1()
    build_s2s3()


if __name__ == "__main__":
    main()
