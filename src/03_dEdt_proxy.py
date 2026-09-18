"""3時間ビン平均した比力学的エネルギー E_J2 から dE/dt (軌道減衰率) 時系列を作る。

対応TODO: #1 (軌道減衰指標の再検討), #4 (Savitzky-Golayフィルタのアーティファクト検証
             — 本スクリプトの --window 引数で感度分析できるようにする),
          #6 (不確かさの定量化 — 欠測を跨がない微分・標準誤差の伝播)

入力:  data/processed/specific_energy.csv          (02_specific_energy.py の出力,
         despike後)
出力:  data/processed/dEdt_3h.csv
       data/figure_data/dEdt_timeseries_full.csv    (同内容。全期間プロット用)

手法:
  1. E_J2_Jkg と a_equiv_m を 3時間ビン (左端基準, pandas "3h") で平均する。
     旧解析 (MT_code, orbit_gps_tle_3.ipynb / estimate_air_density.ipynb) と
     同じ ~3h の代表点間隔に揃えることで、TODO #4 の感度分析や旧proxyとの
     比較 (04_compare_old_proxy.py) がしやすくなる。
     データ欠測でサンプルがないビンは NaN として除外する（区間によっては
     最大9日程度のギャップがある。時間軸は不等間隔になりうる）。
     あわせてビン内標準偏差 (E_J2_Jkg の std) も記録する（#3の元データ）。
  2. dE/dt は中心差分で計算する。ただし np.gradient はそのまま使わない
     （不等間隔でも機械的に重み付けした差分を返してしまい、欠測ギャップを
     跨ぐ点で見かけのスパイクを生む — 例: 2024-07-24/25の18h欠測を跨ぐ
     アーティファクト。docs/2026-07-02_spike_diagnosis.md 改修方針2）。
     代わりに、前後ビンがともにちょうど3h間隔で存在する点のみ中心差分
     dE/dt[i] = (E[i+1] - E[i-1]) / (2*3h) を計算し、それ以外
     （隣接ビン間隔が6hを超える = 前後どちらかのビンが欠測している点）は
     NaN にする。
  3. --window N を指定した場合のみ、微分の前に Savitzky-Golay フィルタ
     (polyorder=1) で E_J2_mean 系列を平滑化する。N は奇数点数のウィンドウ幅
     (旧解析の N=13 ≈ 39時間 に相当する設定も可能)。デフォルト (--window 未指定)
     では平滑化なし = 生の3hビン平均系列をそのまま中心差分する。
     注意: savgol_filter は等間隔仮定なので、ギャップ区間をまたぐ場合は
     厳密には時間軸が歪む (旧解析の実装と同じ制約。詳細は docs 参照)。
     欠測を跨がない中心差分マスク (2.) は --window の有無によらず同じ
     (E_J2_mean の生の時間軸で決まる)。
  4. dE/dt から等価な半長軸減衰率 da/dt へ変換する:
       dE/dt = mu / (2 a^2) * da/dt  ->  da/dt = (2 a^2 / mu) * dE/dt
     a は各ビンの a_equiv_m の平均値を使う。
  5. 不確かさの伝播 (新カラム, #6 / R2 Minor #4):
     ビン内標準偏差 s (E_J2_std) とビン内サンプル数 n から、ビン平均Eの
     標準誤差 se_E = s/sqrt(n) を計算する (n=1のビンは s が定義できないため
     NaN)。dE/dt = (E[i+1]-E[i-1])/(2Δt) の誤差伝播として
       se_dEdt[i] = sqrt(se_E[i-1]^2 + se_E[i+1]^2) / (2Δt)
     (Δt=3h、E[i-1]とE[i+1]は独立とみなす)。2.の欠測マスクと同じ点でのみ
     定義し、それ以外はNaN。--window で平滑化しても se_E 自体は生のビン内
     散らばりに基づくため変化しない (平滑化は不確かさを縮小しない、という
     保守的な扱い)。

実行:
  .venv/Scripts/python src/03_dEdt_proxy.py               # 平滑化なし
  .venv/Scripts/python src/03_dEdt_proxy.py --window 13    # 13点 Savitzky-Golay (約39h)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "specific_energy.csv"
OUT_PROCESSED = ROOT / "data" / "processed" / "dEdt_3h.csv"
OUT_FIG = ROOT / "data" / "figure_data" / "dEdt_timeseries_full.csv"

MU = 3.986004418e14  # GM [m^3/s^2]
BIN = "3h"
BIN_TD = pd.Timedelta(BIN)
SECONDS_PER_DAY = 86400.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--window", type=int, default=None,
        help="Savitzky-Golay平滑化のウィンドウ幅 [点数] (polyorder=1固定, 奇数のみ)。"
             "未指定ならE_J2_mean系列を平滑化せず生の3hビン平均を中心差分する。",
    )
    return p.parse_args()


def gap_aware_center_diff(t: pd.DatetimeIndex, values: np.ndarray,
                           bin_td: pd.Timedelta = BIN_TD) -> np.ndarray:
    """前後ビンがともに bin_td 間隔で存在する点のみ中心差分 (per second) を返す。

    隣接ビン間隔 (どちらか片側でも) が bin_td を超える = 欠測ギャップを
    跨ぐ点は NaN にする (np.gradient のように不等間隔を機械的に重み付け
    しない)。
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    dt_sec = bin_td.total_seconds()
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(np.float64)
    dt_prev = np.full(n, np.nan)
    dt_next = np.full(n, np.nan)
    dt_prev[1:] = gaps
    dt_next[:-1] = gaps
    valid = np.isclose(dt_prev, dt_sec) & np.isclose(dt_next, dt_sec)
    idx = np.where(valid)[0]
    out[idx] = (values[idx + 1] - values[idx - 1]) / (2.0 * dt_sec)
    return out


def main() -> None:
    args = parse_args()
    if args.window is not None and args.window % 2 == 0:
        raise ValueError(f"--window must be odd, got {args.window}")

    df = pd.read_csv(IN, parse_dates=["datetime"]).set_index("datetime")

    binned = df.resample(BIN).agg(
        E_J2_mean=("E_J2_Jkg", "mean"),
        E_J2_std=("E_J2_Jkg", "std"),
        a_mean=("a_equiv_m", "mean"),
        n_samples=("E_J2_Jkg", "count"),
    )
    binned = binned[binned["n_samples"] > 0].copy()

    t = binned.index

    e = binned["E_J2_mean"].to_numpy()
    if args.window is not None:
        e_for_deriv = savgol_filter(e, window_length=args.window, polyorder=1)
    else:
        e_for_deriv = e

    dEdt_Jkg_per_s = gap_aware_center_diff(t, e_for_deriv)
    dEdt_Jkg_per_day = dEdt_Jkg_per_s * SECONDS_PER_DAY

    a = binned["a_mean"].to_numpy()
    dadt_m_per_s = (2.0 * a**2 / MU) * dEdt_Jkg_per_s
    dadt_m_per_day = dadt_m_per_s * SECONDS_PER_DAY

    # --- 不確かさの伝播 (改修3) ---
    n_samples = binned["n_samples"].to_numpy()
    se_E_Jkg = binned["E_J2_std"].to_numpy() / np.sqrt(n_samples)
    se_dEdt_Jkg_per_s = gap_aware_center_diff_se(t, se_E_Jkg)
    se_dEdt_Jkg_per_day = se_dEdt_Jkg_per_s * SECONDS_PER_DAY

    out = pd.DataFrame({
        "datetime": t,
        "E_J2_mean": e,
        "n_samples": n_samples,
        "dEdt_Jkg_per_day": dEdt_Jkg_per_day,
        "dadt_m_per_day": dadt_m_per_day,
        "se_E_Jkg": se_E_Jkg,
        "se_dEdt_Jkg_per_day": se_dEdt_Jkg_per_day,
    })

    OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PROCESSED, index=False)
    print(f"wrote {OUT_PROCESSED} ({len(out)} rows, window={args.window})")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FIG, index=False)
    print(f"wrote {OUT_FIG} ({len(out)} rows)")

    n_nan = int(np.isnan(dEdt_Jkg_per_day).sum())
    print(f"\n=== gap-aware central difference ===")
    print(f"bins with data          : {len(out)} / {(t[-1]-t[0]).days*8} possible 3h bins")
    print(f"dE/dt = NaN (gap-adjacent or endpoint) : {n_nan} / {len(out)} bins")
    print(f"se_dEdt median (valid bins) : "
          f"{np.nanmedian(se_dEdt_Jkg_per_day):.2f} J/kg/day")

    print("\n=== summary ===")
    print(f"dE/dt [J/kg/day]       median={np.nanmedian(dEdt_Jkg_per_day):.2f}  "
          f"p10={np.nanpercentile(dEdt_Jkg_per_day,10):.2f}  "
          f"p90={np.nanpercentile(dEdt_Jkg_per_day,90):.2f}")
    print(f"da/dt [m/day]          median={np.nanmedian(dadt_m_per_day):.2f}  "
          f"p10={np.nanpercentile(dadt_m_per_day,10):.2f}  "
          f"p90={np.nanpercentile(dadt_m_per_day,90):.2f}")


def gap_aware_center_diff_se(t: pd.DatetimeIndex, se_E: np.ndarray,
                              bin_td: pd.Timedelta = BIN_TD) -> np.ndarray:
    """dE/dtの不確かさ se_dEdt = sqrt(se_E[i-1]^2+se_E[i+1]^2)/(2*Δt) (per second)。

    gap_aware_center_diff と同じ有効点マスク (前後ビンがともに bin_td 間隔)
    でのみ定義する。
    """
    n = len(se_E)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    dt_sec = bin_td.total_seconds()
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(np.float64)
    dt_prev = np.full(n, np.nan)
    dt_next = np.full(n, np.nan)
    dt_prev[1:] = gaps
    dt_next[:-1] = gaps
    valid = np.isclose(dt_prev, dt_sec) & np.isclose(dt_next, dt_sec)
    idx = np.where(valid)[0]
    out[idx] = np.sqrt(se_E[idx - 1] ** 2 + se_E[idx + 1] ** 2) / (2.0 * dt_sec)
    return out


if __name__ == "__main__":
    main()
