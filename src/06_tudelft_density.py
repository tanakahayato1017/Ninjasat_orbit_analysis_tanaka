"""TU Delft thermosphere data portalのSwarm A/B POD密度・GRACE-FO加速度計密度を
展開・パースし、衛星ごとに3時間ビン平均密度・日平均密度を計算する。

対応TODO: #5 (独立の熱圏密度データによるdE/dt proxyの比較検証)

入力:
  data/external/tudelft/{SA,SB}_DNS_POD_2024_{04..11}_v02.zip
      Swarm A/B, GPS由来の精密軌道決定(POD)密度。30秒間隔。
      列: Date, Time, TimeSystem(UTC), Altitude(m,GRS80), Lon(deg), Lat(deg),
          LST(h), ArgLat(deg), Density(kg/m3)。
      Swarm Aは高度~440km(参照用)、Swarm Bは高度~510km(NinjaSat ~500-530kmに最近接)。
  data/external/tudelft/GC_DNS_ACC_2024_{04..11}_v02c.zip
      GRACE-FO 1 (GRACE C), 加速度計由来密度。10秒間隔。
      列: Date, Time, TimeSystem(GPS), Altitude(m,GRS80), Lon(deg), Lat(deg),
          LST(h), ArgLat(deg), Density_raw(kg/m3), Density_orbit_mean(kg/m3),
          flag_raw(0=nominal/1=anomalous), flag_mean(同)。
      高度~490-500km。raw density (dens_x) を採用し、flag_raw==0 (nominal) のみ使用。
  取得元URL: https://thermosphere.tudelft.nl/data/data/version_02/
             Swarm_data/ (SA/SB), GRACE-FO_data/ (GC)
  取得日: 2026-07-02 (docs/2026-07-02_tudelft_validation.md 参照)

出力:
  data/processed/tudelft_density_3h.csv
  data/processed/tudelft_density_daily.csv
      共通カラム: datetime, satellite, rho_mean, rho_median, alt_mean, n_samples
      satellite in {"Swarm_A", "Swarm_B", "GRACE-FO"}

手法:
  1. zipファイルをメモリ上で展開し、"#"始まりのヘッダ行を除いて空白区切りで
     パースする (Fortran固定幅フォーマットだが列間は必ず1個以上の空白で
     区切られているため、素朴な空白分割で安全に列を復元できることを
     ヘッダ・データ行を目視確認済み)。
  2. Date+Time列を結合してUTCタイムスタンプにする。GRACE-FOのTime systemは
     "GPS"(UTCとの差はうるう秒起因の一定オフセット、2024年時点で18秒)だが、
     3時間ビン・日次平均に対しては無視できる大きさなので補正しない
     (docsに明記)。
  3. GRACE-FOはflag_raw==0 (nominal) のレコードのみ採用し、dens_x (瞬時値、
     dens_mean=軌道平均値は使わない) を密度として使う。Swarm PODにはQCフラグ
     列が無いため全レコードを採用する。
  4. 緯度によるフィルタは行わない (全緯度平均、タスク仕様通り)。
  5. 衛星ごとに datetime で3時間ビン (pandas "3h", 左端基準) および暦日
     ("1D") でビン平均する。rho_mean/rho_median/alt_mean(m)/n_samples を出す。
     サンプル数0のビンは出力しない。

実行:
  .venv/Scripts/python src/06_tudelft_density.py
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "external" / "tudelft"
OUT_3H = ROOT / "data" / "processed" / "tudelft_density_3h.csv"
OUT_DAILY = ROOT / "data" / "processed" / "tudelft_density_daily.csv"

MONTHS = ["04", "05", "06", "07", "08", "09", "10", "11"]

POD_COLS = ["date", "time", "tsys", "alt_m", "lon", "lat", "lst", "arglat", "rho"]
ACC_COLS = [
    "date", "time", "tsys", "alt_m", "lon", "lat", "lst", "arglat",
    "rho_x", "rho_mean_orbit", "flag_x", "flag_mean",
]

SATELLITES = {
    "Swarm_A": {"prefix": "SA_DNS_POD", "suffix": "v02", "kind": "pod"},
    "Swarm_B": {"prefix": "SB_DNS_POD", "suffix": "v02", "kind": "pod"},
    "GRACE-FO": {"prefix": "GC_DNS_ACC", "suffix": "v02c", "kind": "acc"},
}


def read_zip_month(path: Path, kind: str) -> pd.DataFrame:
    """1ヶ月分のzip (中に1つのtxtファイル) を読み込み、標準化したDataFrameを返す。"""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == 1, f"expected 1 file inside {path}, got {names}"
        with z.open(names[0]) as f:
            raw = f.read().decode("utf-8", errors="replace")

    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("#")]
    buf = io.StringIO("\n".join(lines))

    cols = POD_COLS if kind == "pod" else ACC_COLS
    df = pd.read_csv(buf, sep=r"\s+", header=None, names=cols)

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["datetime"])

    if kind == "pod":
        out = df[["datetime", "alt_m", "rho"]].copy()
    else:
        n_before = len(df)
        df = df[df["flag_x"] == 0.0].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"    {path.name}: dropped {n_dropped}/{n_before} "
                  f"anomalous (flag_x!=0) GRACE-FO records")
        out = df[["datetime", "alt_m", "rho_x"]].rename(columns={"rho_x": "rho"})

    return out


def load_satellite(name: str, spec: dict) -> pd.DataFrame:
    frames = []
    for mm in MONTHS:
        fname = f"{spec['prefix']}_2024_{mm}_{spec['suffix']}.zip"
        path = IN_DIR / fname
        if not path.exists():
            print(f"  WARNING: missing file {path}, skipping this month")
            continue
        df = read_zip_month(path, spec["kind"])
        print(f"    loaded {fname}: {len(df)} rows "
              f"({df['datetime'].min()} .. {df['datetime'].max()})")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no data loaded for satellite {name}")
    full = pd.concat(frames, ignore_index=True).sort_values("datetime")
    n_dup = full.duplicated(subset="datetime").sum()
    if n_dup:
        print(f"  note: {n_dup} duplicate timestamps in {name}, keeping all "
              f"(distinct measurement instants may share nominal timestamp only rarely)")
    full["satellite"] = name
    return full


def bin_stats(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    g = df.set_index("datetime").resample(freq)
    out = g.agg(
        rho_mean=("rho", "mean"),
        rho_median=("rho", "median"),
        alt_mean=("alt_m", "mean"),
        n_samples=("rho", "count"),
    )
    out = out[out["n_samples"] > 0].reset_index()
    return out


def main() -> None:
    all_3h = []
    all_daily = []

    for name, spec in SATELLITES.items():
        print(f"\n=== {name} ({spec['kind']}) ===")
        df = load_satellite(name, spec)

        b3 = bin_stats(df, "3h")
        b3["satellite"] = name
        all_3h.append(b3)

        bd = bin_stats(df, "1D")
        bd["satellite"] = name
        all_daily.append(bd)

        print(f"  raw samples total : {len(df)}")
        print(f"  3h bins           : {len(b3)}")
        print(f"  daily bins        : {len(bd)}")
        print(f"  alt_mean [km]     : {b3['alt_mean'].mean()/1000:.1f} "
              f"(range {b3['alt_mean'].min()/1000:.1f}-{b3['alt_mean'].max()/1000:.1f})")
        print(f"  rho_mean [kg/m3]  : median={np.nanmedian(b3['rho_mean']):.3e}  "
              f"p10={np.nanpercentile(b3['rho_mean'],10):.3e}  "
              f"p90={np.nanpercentile(b3['rho_mean'],90):.3e}")

    out_3h = pd.concat(all_3h, ignore_index=True)[
        ["datetime", "satellite", "rho_mean", "rho_median", "alt_mean", "n_samples"]
    ].sort_values(["satellite", "datetime"])
    out_daily = pd.concat(all_daily, ignore_index=True)[
        ["datetime", "satellite", "rho_mean", "rho_median", "alt_mean", "n_samples"]
    ].sort_values(["satellite", "datetime"])

    OUT_3H.parent.mkdir(parents=True, exist_ok=True)
    out_3h.to_csv(OUT_3H, index=False)
    print(f"\nwrote {OUT_3H} ({len(out_3h)} rows)")

    out_daily.to_csv(OUT_DAILY, index=False)
    print(f"wrote {OUT_DAILY} ({len(out_daily)} rows)")


if __name__ == "__main__":
    main()
