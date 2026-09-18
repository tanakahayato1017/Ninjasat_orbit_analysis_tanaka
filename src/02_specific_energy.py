"""GNSSデータから比力学的エネルギー E を計算する (R1指摘の新しい軌道減衰指標)。

対応TODO: #1 (軌道減衰指標の再検討), #6 (不確かさの定量化 — サンプルレベルdespike)

入力:  data/processed/gps_qc.csv          (01_prepare_gps.py の出力)
出力:  data/processed/specific_energy.csv  (despike後の全サンプルのE)
       data/figure_data/energy_vs_altitude_conservation.csv
         (1軌道内でのE/高度の変動比較 = R1の添付図の再現用データ)

手法 (詳細: docs/2026-07-02_specific_energy_method.md):

  1. 楕円体高 h_ell = gpsAltitudeMeters + gpsGeoid
     (NMEA GGA仕様: altitude はMSL基準、geoid separation を足すと楕円体高)
  2. 測地座標 (lat, lon, h_ell) → ECEF位置 → 地心距離 r, 地心緯度 φ_gc
  3. 慣性系速度の近似:
       v_i² ≈ v_e² + 2·ω·r·v_i·cos(i) − (ω·r·cosφ_gc)²
     v_e = gpsGroundSpeed (ECEF系3次元速度と解釈; 単位km/h→m/s)。
     速度ベクトルが取得できないため、角運動量のz成分を
     h_z ≈ r·v_i·cos(i) (準円軌道近似, i=97.48°はTLEより) で近似。
     v_i について2回の不動点反復で解く (収束は非常に速い)。
  4. E_point = v_i²/2 − μ/r                    (質点ポテンシャル)
     E_J2    = v_i²/2 − μ/r·[1 − J2/2·(Re/r)²·(3sin²φ_gc − 1)]
     R1の指摘どおり、J2項込みのEが軌道内でよく保存されるはず。
  5. サンプルレベルdespike (docs/2026-07-02_spike_diagnosis.md「改修方針」1):
     E_J2_Jkg の13サンプル中心移動中央値 (min_periods=7) からの乖離
     |E_J2 − rolling_median| が、全サンプルに対する大域MAD
     (MAD = median(|E_J2 − rolling_median|)) の10倍を超えるサンプルを
     行ごと除去する (劣化GNSS fixによる単発の速度誤差を源流で除く。
     2024-10-25 04:45:13 UTCの+137 kJ/kg外れ値が典型例。磁気嵐のビン
     スケールの滑らかな変動は誤除去しない設計)。除去件数・日時と
     |dev|をコンソールに出力する。

実行:  .venv/Scripts/python src/02_specific_energy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "gps_qc.csv"
OUT_FULL = ROOT / "data" / "processed" / "specific_energy.csv"
OUT_FIG = ROOT / "data" / "figure_data" / "energy_vs_altitude_conservation.csv"

# WGS84 / 物理定数
A_WGS84 = 6_378_137.0            # 長半径 [m]
F_WGS84 = 1.0 / 298.257223563    # 扁平率
E2 = F_WGS84 * (2.0 - F_WGS84)   # 第一離心率^2
MU = 3.986004418e14              # GM [m^3/s^2]
RE = 6_378_137.0                 # J2項の基準半径 [m]
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5           # 地球自転角速度 [rad/s]
INC_DEG = 97.48                  # NinjaSat軌道傾斜角 (TLEより, ほぼ一定)


DESPIKE_WINDOW = 13          # 中心移動中央値のウィンドウ幅 [サンプル数]
DESPIKE_MIN_PERIODS = 7
DESPIKE_MAD_MULTIPLIER = 10.0


def despike_e_j2(out: pd.DataFrame, col: str = "E_J2_Jkg") -> pd.DataFrame:
    """E_J2 のサンプルレベルdespike (docs/2026-07-02_spike_diagnosis.md 改修方針1)。

    13サンプル中心移動中央値 (min_periods=7) からの乖離が、全サンプルの
    大域MAD (MAD = median(|E_J2 − rolling_median|)) の10倍を超えるサンプルを
    行ごと除去して返す。除去件数・日時をコンソールに出力する。
    """
    roll_median = out[col].rolling(
        window=DESPIKE_WINDOW, center=True, min_periods=DESPIKE_MIN_PERIODS
    ).median()
    dev = (out[col] - roll_median).abs()
    mad = dev.median()
    threshold = DESPIKE_MAD_MULTIPLIER * mad
    mask = dev > threshold

    print(f"\n=== despike on '{col}' (13-sample centered rolling median) ===")
    print(f"global MAD = {mad:.2f} J/kg, threshold = {DESPIKE_MAD_MULTIPLIER}x MAD = "
          f"{threshold:.2f} J/kg")
    print(f"removed {int(mask.sum())} / {len(out)} sample(s):")
    for _, r in out.loc[mask, ["datetime", col]].join(dev[mask].rename("dev")).iterrows():
        print(f"  {r['datetime']}  {col}={r[col]:.2f}  |dev|={r['dev']:.2f} J/kg")

    return out.loc[~mask].reset_index(drop=True)


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """測地座標 (WGS84) → ECEF [m]。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h) * np.sin(lat)
    return x, y, z


def main() -> None:
    df = pd.read_csv(IN, parse_dates=["datetime"])
    if "gpsGroundSpeed" not in df.columns:
        raise SystemExit('gpsGroundSpeed is not part of the public GNSS release. Computing the specific orbital energy requires the velocity column, which needs explicit permission from RIKEN (see data/gnss/README.md). The published 3-hourly product data/processed/dEdt_3h.csv lets every downstream script run without this step.')

    h_ell = df["gpsAltitudeMeters"] + df["gpsGeoid"]
    x, y, z = geodetic_to_ecef(df["gpsLatitude"], df["gpsLongitude"], h_ell)
    r = np.sqrt(x**2 + y**2 + z**2)
    sin_phi_gc = z / r                      # 地心緯度のsin
    cos_phi_gc = np.sqrt(1.0 - sin_phi_gc**2)

    v_e = df["gpsGroundSpeed"] * (1000.0 / 3600.0)   # km/h → m/s

    # 慣性系速度: 不動点反復 (v_i の初期値に v_e を使い2回更新)
    cos_i = np.cos(np.radians(INC_DEG))
    wr = OMEGA_E * r
    v_i = v_e.copy()
    for _ in range(2):
        v_i = np.sqrt(v_e**2 + 2.0 * wr * v_i * cos_i - (wr * cos_phi_gc) ** 2)

    e_point = 0.5 * v_i**2 - MU / r
    u_j2 = -(MU / r) * (1.0 - 0.5 * J2 * (RE / r) ** 2 * (3.0 * sin_phi_gc**2 - 1.0))
    e_j2 = 0.5 * v_i**2 + u_j2

    # 参考: Eから等価な半長軸 a = -mu/(2E) (J2なし定義)
    a_eq = -MU / (2.0 * e_point)

    out = pd.DataFrame({
        "datetime": df["datetime"],
        "gpsUtcTime": df["gpsUtcTime"],
        "lat_deg": df["gpsLatitude"],
        "r_m": r,
        "h_ell_m": h_ell,
        "v_ecef_ms": v_e,
        "v_inertial_ms": v_i,
        "E_point_Jkg": e_point,
        "E_J2_Jkg": e_j2,
        "a_equiv_m": a_eq,
    })

    out = despike_e_j2(out, col="E_J2_Jkg")

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FULL, index=False)
    print(f"\nwrote {OUT_FULL} ({len(out)} rows)")

    # 図データ: 代表2日間について 高度 / E_point / E_J2 の軌道内変動を比較
    # (R1添付図の再現。期間はfigure_dataに含め、描画はplots/側で行う)
    mask = (out["datetime"] >= "2024-07-01") & (out["datetime"] < "2024-07-03")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    out.loc[mask].to_csv(OUT_FIG, index=False)
    print(f"wrote {OUT_FIG} ({int(mask.sum())} rows)")

    # 保存則チェックの要約統計 (1日窓内での標準偏差を比較)
    day = out["datetime"].dt.floor("D")
    grp = out.groupby(day)
    summary = pd.DataFrame({
        "std_h_ell_m": grp["h_ell_m"].std(),
        "std_E_point_Jkg": grp["E_point_Jkg"].std(),
        "std_E_J2_Jkg": grp["E_J2_Jkg"].std(),
    })
    print("\n=== daily within-window std (median over all days) ===")
    print(summary.median().to_string())
    ratio = (summary["std_E_J2_Jkg"] / summary["std_E_point_Jkg"]).median()
    print(f"\nmedian std ratio E_J2/E_point : {ratio:.3f} "
          "(<1 なら J2項込みEの方がよく保存されている)")


if __name__ == "__main__":
    main()
