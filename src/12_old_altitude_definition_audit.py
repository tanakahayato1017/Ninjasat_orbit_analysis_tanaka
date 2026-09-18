r"""旧解析(MT_code)の"GNSS-defined altitude"の実際の定義を生データから逆同定する。

対応TODO: #1 (軌道減衰指標の再検討), #2 (SGP4-GNSS残差の誤差要因の定量化),
          #12 ("GNSS-defined altitude"の定義を明記, R1指摘)

背景:
  R1は"GNSS-defined altitude"の定義が不明瞭と指摘 (GNSS高度は通常WGS84楕円体高)。
  著者が意図していた定義は「地球中心を原点とし地球を完全な球として地心距離から
  一定値を引いたもの (h = r - R0)」だが、実データではgpsAltitudeMeters+gpsGeoid
  (=WGS84楕円体高) がSGP4幾何学的距離|r|の対応する楕円体高とバイアス-22m・
  std108mで一致する一方、素朴な球面解釈 (|r|-一定値 とgpsAltitudeMetersを直接
  比較) だとstd7.5kmに悪化することが既知。

  MT_code/orbit_gps_tle.ipynb (cell 2/3), orbit_gps_tle_debug.ipynb (cell 10),
  orbit_gps_tle_3.ipynb (cell 14) を読み (参照専用、MT_codeは無変更)、旧解析が
  実際に使っていた計算式・推定量を確認した:

    GPS側 (gps_altitude, 1サンプルごと):
      x, y, z = geodetic_to_ecef(gpsLatitude, gpsLongitude, gpsAltitudeMeters)
                (pyproj Transformer EPSG:4326→EPSG:4978, 正しいWGS84楕円体変換)
      h_GPS(t) = sqrt(x^2+y^2+z^2) - 6378.137            # km (定数はWGS84赤道半径)
      *** 入力altitudeが gpsGeoid 未加算の生の gpsAltitudeMeters (測地系/MSL高度)
          であることに注意 (ジオイド高±45m程度がGPS側にのみ混入し、TLE側には
          対応する補正項が無いため Δh に直接系統誤差として入る)。 ***

    TLE側 (tle_altitude, 1サンプルごと), 関数 calculate_orbit_altitude():
      e, r, v = Satrec.twoline2rv(TLE).sgp4(jd, fr)      (TEME座標, km)
      h_TLE(t) = |r| - 6378.137   # 緯度依存の楕円体半径式はコード中でコメントアウトされ未使用
      (TEME→ECEFの回転はノルムを変えないため |r| は回転前後で同じ)

    ビン推定量 (gps_average_altitude / tle_average_altitude), orbit_gps_tle_3.ipynb
    cell14 (このコードが最終的な *_monthly_ext.csv を生成する):
      1. orbital_period = (その月の軌道周期T [分]) × 2         (=2軌道分の窓)
      2. データ先頭からのelapsed_minutesを orbital_period で割った商でビン分割
         (orbit_group = elapsed_minutes // orbital_period, 非重複・逐次)
      3. 各ビンについて、生サンプル数 <= 8 ならスキップ (NaN)
      4. GPS系列・TLE系列それぞれ独立に IQR外れ値除去
         (許容範囲 = [Q1 - 2*IQR, Q3 + 2*IQR]、係数2倍。cell14のコメント
         "# 外れ値範囲を広げる" のとおり2倍を採用しており、以前のドラフト
         セル(9/11)の1.5倍とは異なる。本監査は最終生成コードのcell14に合わせ
         2倍を採用する)
      5. 固定周期の正弦+オフセットをGPS・TLEそれぞれ独立にfit:
           f(x) = A・sin(B_fixed・x + C) + D,  B_fixed = 4π/orbital_period (=2π/T)
           初期値 p0 = [(max-min)/2, π, mean]
      6. gps_average_altitude = D_GPS, tle_average_altitude = D_TLE (poptのオフセット)
      7. altitude_diff = D_TLE - D_GPS

  本スクリプトはこの「IQR外れ値除去 + 固定1/rev正弦フィットのオフセットD」という
  推定量を忠実に再実装し (単純ビン平均ではない)、複数のGPS/TLE高度候補定義に
  適用して旧月別ファイルとのRMSで逆同定する。特定された定義ペア (GPS=h_GPS,
  TLE=h_TLE, いずれもr-6378.137) について、以下3つの誤差源を分離定量化する:
    (α) ジオイド高±45m程度の、GPS側のみへの混入 (TLE側に対応項が無くΔhに直接系統誤差)
    (β) r(t)の真の変動に含まれる2/rev(J2起源、振幅~±数km)成分を、1/rev固定周期の
        正弦フィットが吸収しきれず D に漏れ込ませる残差 (1-harmonic fit と
        2-harmonic fit のDの差として定量化)
    (γ) GPS系列・TLE系列でIQR外れ値除去が独立に(非対称に)効くことによるバイアス
        (各ビンの共通(交差)マスクで再フィットしたΔhと、実際の非対称マスクでの
        Δhとの差として定量化)

入力:
  data/gnss/ninjasat_gnss_2024-04_2024-11.csv
      NinjaSat生GNSSテレメトリ。QCはsrc/01_prepare_gps.pyと同じ基準をこの
      スクリプト内で再実装する (data/processed/は読まない)。
  data/tle/ninjasat_tle_per_day.csv
      TLE(日次間引き)。旧解析同様、対象月(2024-07)の中央付近(7/15前後)の
      1つのTLEを選び、SGP4で対象期間全体に伝播する。
  data/legacy_v1/altitude_diff_monthly_ext/
      gps_tle_average_altitude_fitting_all_severe_2407_monthly_ext.csv
      旧解析の2軌道(~3h)ビン平均ファイル (逆同定のターゲット)。

出力:
  data/figure_data/old_altitude_definition_audit.csv
      旧ファイルとマッチさせた各ビンについて、GPS候補(a,b,c,c')・TLE候補(i,ii)の
      D (IQR除去+1/rev正弦フィット済), 旧ファイル値との残差、α・Δh_old・
      Δh_degeoided・β(net)・γ のビン系列を収録。
  data/figure_data/old_altitude_definition_geometric_2day.csv
      Δh_old(t), α(t), Δh_degeoided(t) の生カデンス(~600s)2日分
      (1周回スケールの可視化用。ビン推定量ではなく生サンプルそのもの)。
  コンソール: 候補ごとのRMS判定サマリ、α・β・γの振幅、sign_audit高周波誤差との相関。

GPS候補 (km):
  (a)  gpsAltitudeMeters                                  生MSL高度そのまま
  (b)  gpsAltitudeMeters + gpsGeoid                        正しいWGS84楕円体高 h_ell
  (c)  |ECEF(lat, lon, h_ell)| - 6378.137                  "ジオイド補正済み"版 r-R0
  (c') |ECEF(lat, lon, gpsAltitudeMeters)| - 6378.137      旧コードの実装 (h_ellでなく
                                                            生のMSL高度を楕円体変換に投入)
TLE候補 (km):
  (i)  |r_SGP4| - 6378.137                                 旧コードの実装 (固定半径)
  (ii) SGP4位置の測地高                                     "正しい"測地高 (TEME z,
       sqrt(x²+y²)から反復計算。経度回転は高度・緯度に無関係なので省略)

実行: .venv/Scripts/python src/12_old_altitude_definition_audit.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import pearsonr, spearmanr
from sgp4.api import Satrec

warnings.filterwarnings("ignore", category=OptimizeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_GPS = ROOT / "data" / "gnss" / "ninjasat_gnss_2024-04_2024-11.csv"
RAW_TLE = ROOT / "data" / "tle" / "ninjasat_tle_per_day.csv"
RAW_OLD = (ROOT / "data" / "legacy_v1" / "altitude_diff_monthly_ext"
           / "gps_tle_average_altitude_fitting_all_severe_2407_monthly_ext.csv")
OUT_BINS = ROOT / "data" / "figure_data" / "old_altitude_definition_audit.csv"
OUT_2DAY = ROOT / "data" / "figure_data" / "old_altitude_definition_geometric_2day.csv"

# WGS84
A_WGS84 = 6378.137               # 長半径(赤道半径) [km]
F_WGS84 = 1.0 / 298.257223563    # 扁平率
E2 = F_WGS84 * (2.0 - F_WGS84)   # 第一離心率^2
R0_OLD = 6378.137                # 旧コードが引いていた固定"地球半径"(=赤道半径) [km]

TARGET_MONTH_MID = pd.Timestamp("2024-07-15", tz="UTC")  # 旧解析の"月中央付近"TLE選定基準
TWO_DAY_WINDOW = ("2024-07-10", "2024-07-12")             # 1周回スケール可視化用
CORE_START = pd.Timestamp("2024-07-01", tz="UTC")
CORE_END = pd.Timestamp("2024-08-01", tz="UTC")
IQR_K = 2.0        # orbit_gps_tle_3.ipynb cell14 の外れ値除去係数 (Q1-2*IQR .. Q3+2*IQR)
MIN_COUNT = 8      # 同cell14: 生サンプル数 <= 8 のビンはスキップ


# --------------------------------------------------------------------------
# QC (src/01_prepare_gps.py と同じ基準をここで再実装。data/processed/は読まない)
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


# --------------------------------------------------------------------------
# 座標変換
# --------------------------------------------------------------------------
def geodetic_to_ecef_km(lat_deg, lon_deg, h_km):
    """測地座標 (WGS84, 高さkm) -> ECEF [km]。MT_code の pyproj EPSG4326->4978と等価。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h_km) * np.cos(lat) * np.cos(lon)
    y = (n + h_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h_km) * np.sin(lat)
    return x, y, z


def ecef_to_geodetic_height_km(x, y, z, n_iter: int = 6):
    """ECEF [km] -> 測地高 [km] (反復法, Bowring型)。経度は不要 (高度・緯度に無関係)。"""
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1.0 - E2))
    for _ in range(n_iter):
        n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - E2 * n / (n + h)))
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return h, np.degrees(lat)


# --------------------------------------------------------------------------
# TLE選定・SGP4伝播
# --------------------------------------------------------------------------
def select_month_mid_tle(target: pd.Timestamp):
    tle = pd.read_csv(RAW_TLE)
    tle["EPOCH"] = pd.to_datetime(tle["EPOCH"], format="mixed")
    tle_epoch_utc = tle["EPOCH"].dt.tz_localize("UTC")
    idx = (tle_epoch_utc - target).abs().idxmin()
    row = tle.loc[idx]
    print(f"selected TLE: EPOCH={row['EPOCH']} (target mid-month {target.date()}), "
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
        print(f"warning: SGP4 returned nonzero error code for {n_bad} samples "
              "(kept as NaN)")
    r = np.asarray(r, dtype=float)
    r[err != 0, :] = np.nan
    return r  # (N, 3) km, TEME frame


# --------------------------------------------------------------------------
# GPS/TLE候補定義 (1サンプルごと)
# --------------------------------------------------------------------------
def compute_gps_candidates(df: pd.DataFrame) -> dict:
    h_msl = df["gpsAltitudeMeters"].to_numpy() / 1e3
    h_ell = h_msl + df["gpsGeoid"].to_numpy() / 1e3
    lat = df["gpsLatitude"].to_numpy()
    lon = df["gpsLongitude"].to_numpy()

    x_b, y_b, z_b = geodetic_to_ecef_km(lat, lon, h_ell)
    r_b = np.sqrt(x_b ** 2 + y_b ** 2 + z_b ** 2)

    x_raw, y_raw, z_raw = geodetic_to_ecef_km(lat, lon, h_msl)
    r_raw = np.sqrt(x_raw ** 2 + y_raw ** 2 + z_raw ** 2)

    return {
        "a_msl_raw": h_msl,
        "b_ellipsoidal_h_ell": h_ell,
        "c_r_minus_R0_geoid_corrected": r_b - R0_OLD,
        "cprime_r_minus_R0_asused": r_raw - R0_OLD,
    }


def compute_tle_candidates(r_teme: np.ndarray) -> dict:
    r_norm = np.linalg.norm(r_teme, axis=1)
    h_geodetic, lat_deg = ecef_to_geodetic_height_km(
        r_teme[:, 0], r_teme[:, 1], r_teme[:, 2])
    return {
        "i_r_minus_R0_asused": r_norm - R0_OLD,
        "ii_geodetic_height_correct": h_geodetic,
        "tle_lat_deg": lat_deg,
    }


# --------------------------------------------------------------------------
# 旧コードの推定量: IQR外れ値除去 + 固定1/rev正弦フィットのオフセットD
# (orbit_gps_tle_3.ipynb cell14 の忠実な再実装)
# --------------------------------------------------------------------------
def bin_indices(elapsed_min: np.ndarray, orbital_period_min: float, min_count: int = MIN_COUNT):
    """データ先頭からのelapsed_minutesを2軌道幅の窓で非重複・逐次分割。
    各ビンについて (生サンプル数<=min_countならNone) の生インデックス配列を返す。"""
    n_bins = int(np.ceil(elapsed_min.max() / orbital_period_min)) if len(elapsed_min) else 0
    out = []
    for i in range(n_bins):
        lo, hi = i * orbital_period_min, (i + 1) * orbital_period_min
        idx = np.where((elapsed_min >= lo) & (elapsed_min < hi))[0]
        out.append(idx if len(idx) > min_count else None)
    return out


def iqr_keep_mask(y: np.ndarray, k: float = IQR_K) -> np.ndarray:
    """ビン内の値yに対するIQR外れ値除去マスク ([Q1-k*IQR, Q3+k*IQR] を保持)。"""
    finite = np.isfinite(y)
    if finite.sum() < 4:
        return finite
    q1, q3 = np.nanpercentile(y[finite], [25, 75])
    iqr = q3 - q1
    return finite & (y >= q1 - k * iqr) & (y <= q3 + k * iqr)


def fit_offset(x: np.ndarray, y: np.ndarray, B: float, n_harmonics: int = 1) -> float:
    """固定周期(B=2π/T, n_harmonics=1) または 1/rev+2/rev(n_harmonics=2) の
    正弦+オフセットをfitし、オフセットDを返す (失敗時はnan)。"""
    n_params = 3 if n_harmonics == 1 else 5
    if len(x) < n_params + 2:
        return np.nan
    amp0 = (np.nanmax(y) - np.nanmin(y)) / 2.0
    mean0 = np.nanmean(y)
    try:
        if n_harmonics == 1:
            def f(xx, A, C, D):
                return A * np.sin(B * xx + C) + D
            popt, _ = curve_fit(f, x, y, p0=[amp0, np.pi, mean0], maxfev=5000)
            return popt[2]
        else:
            def f(xx, A1, C1, A2, C2, D):
                return A1 * np.sin(B * xx + C1) + A2 * np.sin(2.0 * B * xx + C2) + D
            popt, _ = curve_fit(f, x, y, p0=[amp0, np.pi, amp0 / 3.0, np.pi, mean0],
                                 maxfev=5000)
            return popt[4]
    except Exception:
        return np.nan


def main() -> None:
    old = pd.read_csv(RAW_OLD)
    old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
    old = old.dropna(subset=["timestamp", "gps_average_altitude",
                              "tle_average_altitude", "altitude_diff"]).reset_index(drop=True)
    print(f"old file bins: {len(old)}  "
          f"({old['timestamp'].min()} .. {old['timestamp'].max()})")

    t_min = old["timestamp"].min() - pd.Timedelta(days=1)
    t_max = old["timestamp"].max() + pd.Timedelta(days=1)

    gps = load_and_qc_gps(t_min, t_max)
    print(f"QC'd raw GPS samples in window: {len(gps)}")

    tle1, tle2, mean_motion = select_month_mid_tle(TARGET_MONTH_MID)
    orbit_period_min = 1440.0 / mean_motion       # 単一周回長T [分]
    orbital_period = 2.0 * orbit_period_min       # 旧コードの"orbital_period" (2軌道幅) [分]
    B_fixed = 4.0 * np.pi / orbital_period        # = 2π/T
    print(f"single-orbit period T = {orbit_period_min:.3f} min, "
          f"2-orbit bin width = {orbital_period:.3f} min (matches orbit_gps_tle_3.ipynb cell14)")

    gps_cands = compute_gps_candidates(gps)
    r_teme = propagate_sgp4_teme(tle1, tle2, pd.DatetimeIndex(gps["datetime"]))
    tle_cands = compute_tle_candidates(r_teme)

    times = pd.DatetimeIndex(gps["datetime"])
    elapsed_min = (times - times.min()).total_seconds().to_numpy() / 60.0
    idx_list = bin_indices(elapsed_min, orbital_period)
    print(f"n candidate bins (this script's own t0 reference): {len(idx_list)} "
          f"({sum(b is not None for b in idx_list)} with n_raw>{MIN_COUNT})")

    all_series = {**{f"gps_{k}": v for k, v in gps_cands.items()},
                  **{f"tle_{k}": v for k, v in tle_cands.items() if k != "tle_lat_deg"}}
    gps_names = [f"gps_{k}" for k in gps_cands]
    tle_names = [f"tle_{k}" for k in tle_cands if k != "tle_lat_deg"]

    rows = []
    for idx in idx_list:
        row = {}
        if idx is None:
            row["center_time"] = pd.NaT
            row["n_raw"] = 0
            for name in all_series:
                row[name] = np.nan
            row["D2_gps_cprime"] = np.nan
            row["D2_tle_i"] = np.nan
            row["D_common_gps_cprime"] = np.nan
            row["D_common_tle_i"] = np.nan
            rows.append(row)
            continue

        x = elapsed_min[idx]
        row["center_time"] = times[idx].mean()
        row["n_raw"] = len(idx)

        keep_masks = {}
        for name, arr in all_series.items():
            y = arr[idx]
            keep = iqr_keep_mask(y)
            keep_masks[name] = keep
            row[name] = fit_offset(x[keep], y[keep], B_fixed, n_harmonics=1)

        # β: 1/rev vs 1/rev+2/rev フィットのオフセット差 (own mask, GPS(c')・TLE(i)のみ)
        y_cprime = gps_cands["cprime_r_minus_R0_asused"][idx]
        y_i = tle_cands["i_r_minus_R0_asused"][idx]
        km_c = keep_masks["gps_cprime_r_minus_R0_asused"]
        km_i = keep_masks["tle_i_r_minus_R0_asused"]
        row["D2_gps_cprime"] = fit_offset(x[km_c], y_cprime[km_c], B_fixed, n_harmonics=2)
        row["D2_tle_i"] = fit_offset(x[km_i], y_i[km_i], B_fixed, n_harmonics=2)

        # γ: GPS(c')・TLE(i)の共通(交差)マスクで再フィット (対称除去した場合の値)
        common = km_c & km_i
        row["D_common_gps_cprime"] = fit_offset(x[common], y_cprime[common], B_fixed, n_harmonics=1)
        row["D_common_tle_i"] = fit_offset(x[common], y_i[common], B_fixed, n_harmonics=1)

        rows.append(row)

    mine = pd.DataFrame(rows).dropna(subset=["center_time"]).reset_index(drop=True)
    # dtype統一 (times起源のcenter_timeはdatetime64[s]になりうる。old["timestamp"]は
    # datetime64[ns]なのでmerge_asofのため揃える)
    mine["center_time"] = pd.to_datetime(mine["center_time"], utc=True).astype("datetime64[ns, UTC]")
    mine = mine.sort_values("center_time").reset_index(drop=True)
    print(f"fitted bins with valid center_time: {len(mine)}")

    # ---------------- α, Δh_old, Δh_degeoided, β(net), γ (ビンレベル, own t0) ----------------
    # (merge_asofより前に計算しないと出力CSV/相関解析にこれらの列が乗らないので注意)
    mine["alpha_binned"] = (mine["gps_c_r_minus_R0_geoid_corrected"]
                             - mine["gps_cprime_r_minus_R0_asused"])
    mine["delta_h_old_binned"] = (mine["tle_i_r_minus_R0_asused"]
                                   - mine["gps_cprime_r_minus_R0_asused"])
    mine["delta_h_degeoided_binned"] = (mine["tle_i_r_minus_R0_asused"]
                                         - mine["gps_c_r_minus_R0_geoid_corrected"])
    beta_gps = mine["gps_cprime_r_minus_R0_asused"] - mine["D2_gps_cprime"]
    beta_tle = mine["tle_i_r_minus_R0_asused"] - mine["D2_tle_i"]
    mine["beta_binned"] = beta_tle - beta_gps   # 2/rev漏れ込みがΔh_oldに与える正味バイアス
    delta_asym = mine["delta_h_old_binned"]
    delta_sym = mine["D_common_tle_i"] - mine["D_common_gps_cprime"]
    mine["gamma_binned"] = delta_asym - delta_sym  # 非対称IQR除去がΔh_oldに与える正味バイアス

    # ---------------- 旧ファイルと最近傍タイムスタンプでマッチング ----------------
    old_sorted = old.sort_values("timestamp").reset_index(drop=True)
    tol = pd.Timedelta(minutes=orbital_period / 2.0)
    merged = pd.merge_asof(old_sorted, mine, left_on="timestamp", right_on="center_time",
                            direction="nearest", tolerance=tol)
    print(f"matched to archived bins: {merged['center_time'].notna().sum()} / {len(old_sorted)} "
          f"(tolerance ±{tol})")

    # ---------------- RMS判定 ----------------
    gps_rms = {}
    print("\n=== GPS候補 (IQR除去+1/rev正弦フィットD): 旧gps_average_altitudeとのRMS ===")
    for name in gps_names:
        ok = merged[name].notna() & merged["gps_average_altitude"].notna()
        resid = merged.loc[ok, name] - merged.loc[ok, "gps_average_altitude"]
        rms = np.sqrt(np.mean(resid ** 2))
        gps_rms[name] = rms
        print(f"  {name:38s}  RMS = {rms*1000:9.2f} m   (n_bins={int(ok.sum())})")
    best_gps = min(gps_rms, key=gps_rms.get)

    tle_rms = {}
    print("\n=== TLE候補 (IQR除去+1/rev正弦フィットD): 旧tle_average_altitudeとのRMS ===")
    for name in tle_names:
        ok = merged[name].notna() & merged["tle_average_altitude"].notna()
        resid = merged.loc[ok, name] - merged.loc[ok, "tle_average_altitude"]
        rms = np.sqrt(np.mean(resid ** 2))
        tle_rms[name] = rms
        print(f"  {name:38s}  RMS = {rms*1000:9.2f} m   (n_bins={int(ok.sum())})")
    best_tle = min(tle_rms, key=tle_rms.get)

    print(f"\n最良一致: GPS={best_gps} (RMS={gps_rms[best_gps]*1000:.2f} m), "
          f"TLE={best_tle} (RMS={tle_rms[best_tle]*1000:.2f} m)")

    core_mask = (mine["center_time"] >= CORE_START) & (mine["center_time"] < CORE_END)

    def report_term(col: str, label: str) -> pd.Series:
        s = mine.loc[core_mask, col].dropna()
        print(f"  {label:28s} mean={s.mean()*1000:8.2f} m  std={s.std()*1000:8.2f} m  "
              f"n={len(s)}")
        return s

    print(f"\n=== ビンレベル誤差項 (own t0基準の2軌道ビン, コア期間) ===")
    report_term("alpha_binned", "alpha (geoid)")
    report_term("delta_h_old_binned", "Delta h_old")
    report_term("delta_h_degeoided_binned", "Delta h_degeoided")
    report_term("beta_binned", "beta (2/rev leak, net)")
    report_term("gamma_binned", "gamma (asym. IQR, net)")
    n_gamma_nonzero = int((mine.loc[core_mask, "gamma_binned"].abs() > 1e-9).sum())
    n_gamma_total = int(mine.loc[core_mask, "gamma_binned"].notna().sum())
    print(f"  -> gamma is exactly nonzero in {n_gamma_nonzero}/{n_gamma_total} core bins "
          f"(IQR_K={IQR_K} is loose enough that GPS/TLE outlier masks coincide almost always)")

    # ---------------- sign_audit流の高周波残差との相関 ----------------
    def sign_audit_highpass(series: pd.Series) -> pd.Series:
        grid = series.resample("3h").mean()
        roll = grid.rolling(9, center=True, min_periods=5).mean()
        return grid - roll

    old_series = old.set_index("timestamp")["altitude_diff"].sort_index()
    old_hp = sign_audit_highpass(old_series)

    mine_ts = mine.set_index("center_time")
    print(f"\n=== sign_audit高周波残差 (24h rolling mean除去, 3hグリッド) との相関 "
          f"(コア期間) ===")
    corr_results = {}
    for col, label in [("alpha_binned", "alpha (geoid)"),
                        ("delta_h_degeoided_binned", "Delta h_degeoided"),
                        ("beta_binned", "beta (2/rev leak)"),
                        ("gamma_binned", "gamma (asym. IQR)")]:
        term_hp = sign_audit_highpass(mine_ts[col].dropna())
        merged_hp = pd.DataFrame({"old_hp": old_hp, "term_hp": term_hp}).dropna()
        merged_hp_core = merged_hp[(merged_hp.index >= CORE_START) & (merged_hp.index < CORE_END)]
        if len(merged_hp_core) >= 5 and merged_hp_core["term_hp"].std() > 1e-9:
            rp, pp = pearsonr(merged_hp_core["term_hp"], merged_hp_core["old_hp"])
            rs, ps = spearmanr(merged_hp_core["term_hp"], merged_hp_core["old_hp"])
            corr_results[col] = (rp, rs)
            print(f"  {label:22s}: Pearson r={rp:+.4f} (p={pp:.3g}), "
                  f"Spearman rho={rs:+.4f} (p={ps:.3g}), n={len(merged_hp_core)}")
        else:
            corr_results[col] = (np.nan, np.nan)
            print(f"  {label:22s}: insufficient overlapping bins or term is ~constant (n={len(merged_hp_core)})")

    # ---------------- 生カデンス: α, Δh_old, Δh_degeoided (単一周回内変動の把握用) ----------------
    delta_h_old_raw = tle_cands["i_r_minus_R0_asused"] - gps_cands["cprime_r_minus_R0_asused"]
    alpha_raw = (gps_cands["c_r_minus_R0_geoid_corrected"]
                 - gps_cands["cprime_r_minus_R0_asused"])
    delta_h_degeoided_raw = (tle_cands["i_r_minus_R0_asused"]
                              - gps_cands["c_r_minus_R0_geoid_corrected"])
    core_raw = (times >= CORE_START) & (times < CORE_END)

    orbit_group_raw = (elapsed_min // orbit_period_min).astype(int)
    df_orb = pd.DataFrame({"orbit_group": orbit_group_raw, "alpha": alpha_raw,
                            "delta_degeoided": delta_h_degeoided_raw, "core": core_raw})
    per_orbit = df_orb[df_orb["core"]].groupby("orbit_group").agg(
        alpha_std=("alpha", "std"), alpha_ptp=("alpha", lambda s: s.max() - s.min()),
        deg_std=("delta_degeoided", "std"), deg_ptp=("delta_degeoided", lambda s: s.max() - s.min()),
        count=("alpha", "count"))
    per_orbit = per_orbit[per_orbit["count"] >= 5]
    print(f"\n=== 単一周回(~T={orbit_period_min:.1f}min)内変動幅 (生カデンス, n_orbits={len(per_orbit)}, "
          f"コア期間) ===")
    print(f"  alpha (geoid)     : median std = {per_orbit['alpha_std'].median()*1000:7.2f} m, "
          f"median ptp = {per_orbit['alpha_ptp'].median()*1000:7.2f} m")
    print(f"  Delta h_degeoided : median std = {per_orbit['deg_std'].median()*1000:7.2f} m, "
          f"median ptp = {per_orbit['deg_ptp'].median()*1000:7.2f} m "
          f"(r(t)自体の1/rev+2/rev振動の目安)")

    # ---------------- 出力: ビンごとの再構成値・誤差項 ----------------
    out_cols = ["timestamp", "gps_average_altitude", "tle_average_altitude", "altitude_diff",
                "center_time", "n_raw"] + gps_names + tle_names + [
        "D2_gps_cprime", "D2_tle_i", "D_common_gps_cprime", "D_common_tle_i",
        "alpha_binned", "delta_h_old_binned", "delta_h_degeoided_binned",
        "beta_binned", "gamma_binned"]
    out_df = merged.reindex(columns=[c for c in out_cols if c in merged.columns]).rename(
        columns={"gps_average_altitude": "old_gps_average_altitude",
                 "tle_average_altitude": "old_tle_average_altitude",
                 "altitude_diff": "old_altitude_diff"})
    OUT_BINS.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_BINS, index=False)
    print(f"\nwrote {OUT_BINS} ({len(out_df)} rows)")

    # ---------------- 出力: 生カデンス2日分 (1周回スケール可視化用) ----------------
    win_mask = (times >= pd.Timestamp(TWO_DAY_WINDOW[0], tz="UTC")) & \
               (times < pd.Timestamp(TWO_DAY_WINDOW[1], tz="UTC"))
    two_day = pd.DataFrame({
        "datetime": times[win_mask],
        "gps_lat_deg": gps.loc[win_mask, "gpsLatitude"].to_numpy(),
        "tle_lat_deg": tle_cands["tle_lat_deg"][win_mask],
        "delta_h_old": delta_h_old_raw[win_mask],
        "alpha_geoid": alpha_raw[win_mask],
        "delta_h_degeoided": delta_h_degeoided_raw[win_mask],
    })
    two_day.to_csv(OUT_2DAY, index=False)
    print(f"wrote {OUT_2DAY} ({len(two_day)} rows, {TWO_DAY_WINDOW[0]}..{TWO_DAY_WINDOW[1]})")

    print("\n=== 判定サマリ ===")
    print("旧解析の実際の定義 (MT_codeソース確認 + IQR除去+1/rev正弦フィットDでのRMS最小の組):")
    print("  GPS: (c') |ECEF(lat,lon,gpsAltitudeMeters)| - 6378.137  "
          "(geoid未加算のraw MSL高度を楕円体変換に投入)")
    print("  TLE: (i)  |r_SGP4(TEME)| - 6378.137  (緯度依存の楕円体半径式は未使用)")
    print("  推定量: D = IQR外れ値除去(2×IQR) + 固定1/rev正弦フィットのオフセット "
          "(2軌道窓, orbit_gps_tle_3.ipynb cell14)")
    print(f"  => h_GNSS = |r_ECEF(phi, lambda, h_MSL)| - 6378.137 km,  "
          f"h_bar(bin) = D (1/rev正弦+オフセットフィット, 2軌道窓, IQR外れ値除去付き)")
    a_stats = mine.loc[core_mask, "alpha_binned"].dropna()
    b_stats = mine.loc[core_mask, "beta_binned"].dropna()
    g_stats = mine.loc[core_mask, "gamma_binned"].dropna()
    print(f"alpha(ジオイド, ビン): std={a_stats.std()*1000:.1f} m, "
          f"beta(2/rev漏れ込み, ビン): std={b_stats.std()*1000:.1f} m, "
          f"gamma(非対称IQR, ビン): std={g_stats.std()*1000:.1f} m")
    for col, label in [("alpha_binned", "alpha"), ("delta_h_degeoided_binned", "Dh_degeoided"),
                        ("beta_binned", "beta"), ("gamma_binned", "gamma")]:
        rp, rs = corr_results[col]
        print(f"  {label} vs sign_audit高周波誤差: Pearson r={rp:+.3f}, Spearman rho={rs:+.3f}")


if __name__ == "__main__":
    main()
