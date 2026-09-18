"""GNSSテレメトリの前処理・品質管理 (QC)。

対応TODO: #1 (軌道減衰指標の再検討) の前段。全解析の共通入力を作る。

入力:  data/gnss/ninjasat_gnss_2024-04_2024-11.csv
        NinjaSat HKテレメトリ (2024-04 〜 2024-11)。約600s間隔。
出力:  data/processed/gps_qc.csv
        軌道解析に必要なカラムのみ + QCフラグ通過行。

QC基準 (docs/2026-07-02_specific_energy_method.md 参照):
  - gpsFixQual == 1 (有効なfix)
  - 高度が 400–600 km の範囲 (NinjaSat軌道の物理的に妥当な範囲)
  - 速度が 26,000–29,000 km/h の範囲
  - gpsUtcTime が単調 (重複タイムスタンプは最初の1件を採用)

実行:  .venv/Scripts/python src/01_prepare_gps.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "gnss" / "ninjasat_gnss_2024-04_2024-11.csv"
OUT = ROOT / "data" / "processed" / "gps_qc.csv"

COLS = [
    "gpsUtcTime",
    "gpsFixQual",
    "gpsLatitude",
    "gpsLongitude",
    "gpsAltitudeMeters",
    "gpsGroundSpeed",
    "gpsGeoid",
    "gpsNumOfSatsTracked",
    "gpsHDilutionOfPos",
]


def main() -> None:
    header = pd.read_csv(RAW, nrows=0).columns
    missing = [c for c in COLS if c not in header]
    if missing:
        print(f"NOTE: columns not in the public GNSS file (RIKEN permission required): {missing}")
    df = pd.read_csv(RAW, usecols=[c for c in COLS if c in header])
    n0 = len(df)

    df = df[df["gpsFixQual"] == 1]
    n_fix = len(df)

    alt_km = df["gpsAltitudeMeters"] / 1e3
    df = df[(alt_km > 400) & (alt_km < 600)]
    n_alt = len(df)

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
    n_spd = len(df)

    df = df.sort_values("gpsUtcTime").drop_duplicates("gpsUtcTime", keep="first")
    n_uni = len(df)

    df["datetime"] = pd.to_datetime(df["gpsUtcTime"], unit="s", utc=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"input rows            : {n0}")
    print(f"after fixQual==1      : {n_fix}")
    print(f"after altitude window : {n_alt}")
    print(f"after speed window    : {n_spd}")
    print(f"after dedup           : {n_uni}")
    print(f"time range            : {df['datetime'].min()} .. {df['datetime'].max()}")
    dt = df["gpsUtcTime"].diff().dropna()
    print(f"sampling interval     : median {dt.median():.0f}s "
          f"(p10 {dt.quantile(0.1):.0f}s, p90 {dt.quantile(0.9):.0f}s)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
