r"""高度解釈の判定実験の形式化 (図2: altitude_semantics)。

対応TODO: #1, #12 ("GNSS-defined altitude"の定義を明記, R1指摘)

背景:
  docs/2026-07-02_specific_energy_method.md では `gpsAltitudeMeters` を
  「NMEA GGA仕様のMSL高度」と解釈し、`gpsGeoid` を足した楕円体高
  h_ell = alt + geoid をECEF変換に投入している (02_specific_energy.py)。
  この解釈は事前の突き合わせ作業でSGP4幾何と良く一致することが分かって
  いたが、それを再現可能な形式的検証として残す図がなかった。本スクリプトは
  2024-07-01のテレメトリ1日分について、3つの高度解釈仮説をSGP4の|r|と
  直接比較する:

    H1 (採用した解釈, 楕円体高):
      |ECEF(lat, lon, alt+geoid)|  vs  |r_SGP4|
    H2 (素朴な球面解釈, WGS84赤道半径):
      alt + 6378.137 km            vs  |r_SGP4|
    H3 (素朴な球面解釈, 地球平均半径):
      alt + 6371 km                 vs  |r_SGP4|

  H2/H3は「altをそのまま地心距離のオフセットとみなす」解釈で、緯度による
  地心距離の変化 (WGS84扁平, 赤道~6378km / 極~6357km) を全く考慮しないため、
  緯度に強く相関する大きな残差が出ることが期待される。H1はWGS84楕円体変換を
  経由するため、この緯度依存項が正しく吸収される。

入力:
  data/processed/gps_qc.csv                (2024-07-01分を抽出)
  data/tle/ninjasat_tle_per_day.csv         (7/1に最も近いTLEを選択)

出力:
  data/figure_data/altitude_semantics.csv
    列: datetime, lat_deg, resid_H1_m, resid_H2_km, resid_H3_km
  data/figure_data/altitude_semantics_stats.csv
    列: hypothesis, unit, bias, std, n

実行: .venv/Scripts/python src/17_altitude_semantics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parents[1]
IN_GPS = ROOT / "data" / "processed" / "gps_qc.csv"
IN_TLE = ROOT / "data" / "tle" / "ninjasat_tle_per_day.csv"
OUT_TS = ROOT / "data" / "figure_data" / "altitude_semantics.csv"
OUT_STATS = ROOT / "data" / "figure_data" / "altitude_semantics_stats.csv"

TARGET_DATE = pd.Timestamp("2024-07-01", tz="UTC")

# WGS84
A_WGS84 = 6_378_137.0            # 長半径(赤道半径) [m]
F_WGS84 = 1.0 / 298.257223563
E2 = F_WGS84 * (2.0 - F_WGS84)

R_EQ_KM = 6378.137     # H2: WGS84赤道半径
R_MEAN_KM = 6371.0     # H3: 地球平均半径 (IUGG)


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """測地座標 (WGS84, 高さm) -> ECEF [m]。02_specific_energy.pyと同じ変換式。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h_m) * np.sin(lat)
    return x, y, z


def select_nearest_tle(target: pd.Timestamp):
    tle = pd.read_csv(IN_TLE)
    tle["EPOCH"] = pd.to_datetime(tle["EPOCH"], format="mixed")
    tle_epoch_utc = tle["EPOCH"].dt.tz_localize("UTC")
    idx = (tle_epoch_utc - target).abs().idxmin()
    row = tle.loc[idx]
    print(f"selected TLE: EPOCH={row['EPOCH']} (target={target.date()}), "
          f"delta={abs(tle_epoch_utc[idx]-target)}")
    return row["TLE_LINE1"], row["TLE_LINE2"]


def propagate_sgp4_norm_km(tle_line1: str, tle_line2: str, times_utc: pd.DatetimeIndex):
    """SGP4伝播した|r| [km] (TEME frame)。回転はノルムを保つのでECEF変換は不要
    (12_old_altitude_definition_audit.py と同じ扱い)。"""
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd_total = times_utc.to_julian_date().to_numpy()
    jd = np.floor(jd_total)
    fr = jd_total - jd
    err, r, _v = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        print(f"warning: SGP4 nonzero error for {int(np.sum(err != 0))} samples")
    r = np.asarray(r, dtype=float)
    r[err != 0, :] = np.nan
    return np.linalg.norm(r, axis=1)


def main() -> None:
    gps = pd.read_csv(IN_GPS, parse_dates=["datetime"])
    day_mask = (gps["datetime"] >= TARGET_DATE) & (gps["datetime"] < TARGET_DATE + pd.Timedelta(days=1))
    gps = gps.loc[day_mask].sort_values("datetime").reset_index(drop=True)
    print(f"GPS QC samples on {TARGET_DATE.date()}: {len(gps)}")

    tle1, tle2 = select_nearest_tle(TARGET_DATE + pd.Timedelta(hours=12))
    r_sgp4_km = propagate_sgp4_norm_km(tle1, tle2, pd.DatetimeIndex(gps["datetime"]))

    # H1: 楕円体高解釈 |ECEF(lat, lon, alt+geoid)|
    h_ell_m = gps["gpsAltitudeMeters"].to_numpy() + gps["gpsGeoid"].to_numpy()
    x, y, z = geodetic_to_ecef(gps["gpsLatitude"].to_numpy(), gps["gpsLongitude"].to_numpy(), h_ell_m)
    r_h1_m = np.sqrt(x**2 + y**2 + z**2)
    resid_h1_m = r_h1_m - r_sgp4_km * 1000.0

    # H2 / H3: 素朴な球面解釈 (altをそのまま地心距離オフセットとみなす)
    alt_km = gps["gpsAltitudeMeters"].to_numpy() / 1000.0
    r_h2_km = alt_km + R_EQ_KM
    r_h3_km = alt_km + R_MEAN_KM
    resid_h2_km = r_h2_km - r_sgp4_km
    resid_h3_km = r_h3_km - r_sgp4_km

    out = pd.DataFrame({
        "datetime": gps["datetime"],
        "lat_deg": gps["gpsLatitude"],
        "resid_H1_m": resid_h1_m,
        "resid_H2_km": resid_h2_km,
        "resid_H3_km": resid_h3_km,
    })
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TS, index=False)
    print(f"\nwrote {OUT_TS} ({len(out)} rows)")

    stats_rows = []
    for name, col, unit in [
        ("H1_ellipsoidal", "resid_H1_m", "m"),
        ("H2_alt_plus_Req", "resid_H2_km", "km"),
        ("H3_alt_plus_Rmean", "resid_H3_km", "km"),
    ]:
        s = out[col].dropna()
        bias, std = s.mean(), s.std()
        stats_rows.append({"hypothesis": name, "unit": unit, "bias": bias, "std": std, "n": len(s)})
        print(f"{name:20s} bias={bias:+9.3f} {unit}  std={std:8.3f} {unit}  (n={len(s)})")

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT_STATS, index=False)
    print(f"\nwrote {OUT_STATS}")


if __name__ == "__main__":
    main()
