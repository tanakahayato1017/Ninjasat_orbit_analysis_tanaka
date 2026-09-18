"""ミッション全期間のE_J2時系列 (図4: E_full_timeseries) の図データ作成。

対応TODO: #1 (軌道減衰指標の再検討) の全体像可視化

背景:
  specific_energy_method / dEdt_proxy / spike_diagnosis の各文書は個々の
  手法・診断を扱うが、「E_J2が全期間を通して単調に減衰し、太陽活動の上昇
  とともに減衰が加速する」という基盤パイプライン全体のストーリーを1枚で
  示す図がなかった。本スクリプトは despike後の全期間 E_J2 (data/processed/
  specific_energy.csv) を3hビン平均し、同じビンの a_equiv_m (E_pointから
  逆算した等価半長軸) も平均する。Kp>=6 (地磁気嵐, 例: Gannon storm
  2024-05-10/11) の3hビンを data/processed/geomag_3h.csv から特定し、
  連続区間としてまとめる (plot_full_timeseries_euv.py の storm_intervals()
  と同じロジックをここで独立に再実装)。

入力:
  data/processed/specific_energy.csv  (02_specific_energy.py の出力, despike後)
  data/processed/geomag_3h.csv        (05_geomagnetic_indices.py の出力)

出力:
  data/figure_data/E_full_timeseries.csv
    列: datetime, E_J2_mean_Jkg, a_equiv_mean_km, n_samples, kp, storm_kp6
  data/figure_data/E_full_timeseries_storm_intervals.csv
    列: start, end  (Kp>=6の連続区間, 縦帯描画用)

実行: .venv/Scripts/python src/17_E_full_timeseries_figdata.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_E = ROOT / "data" / "processed" / "specific_energy.csv"
IN_GEOMAG = ROOT / "data" / "processed" / "geomag_3h.csv"
OUT_TS = ROOT / "data" / "figure_data" / "E_full_timeseries.csv"
OUT_STORMS = ROOT / "data" / "figure_data" / "E_full_timeseries_storm_intervals.csv"

KP_STORM_THRESHOLD = 6.0
BIN = "3h"


def storm_intervals(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """連続する Kp>=6 の3hビンをまとめて [start, end) 区間のリストにする
    (plot_full_timeseries_euv.py の同名関数と同じロジックの独立再実装)。"""
    storm_times = df.loc[df["storm_kp6"].fillna(False), "datetime"].sort_values().to_numpy()
    if len(storm_times) == 0:
        return []
    intervals = []
    start = storm_times[0]
    prev = storm_times[0]
    step = np.timedelta64(3, "h")
    for t in storm_times[1:]:
        if t - prev > step:
            intervals.append((pd.Timestamp(start), pd.Timestamp(prev) + pd.Timedelta(BIN)))
            start = t
        prev = t
    intervals.append((pd.Timestamp(start), pd.Timestamp(prev) + pd.Timedelta(BIN)))
    return intervals


def main() -> None:
    e = pd.read_csv(IN_E, parse_dates=["datetime"])
    e["datetime"] = e["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    e = e.set_index("datetime").sort_index()

    grp = e[["E_J2_Jkg", "a_equiv_m"]].resample(BIN)
    binned = grp.mean()
    binned["n_samples"] = grp.size()
    binned = binned.rename(columns={"E_J2_Jkg": "E_J2_mean_Jkg"})
    binned["a_equiv_mean_km"] = binned.pop("a_equiv_m") / 1000.0
    binned = binned.reset_index()
    binned = binned[binned["n_samples"] > 0].reset_index(drop=True)
    print(f"3h-binned E_J2: {len(binned)} bins with data "
          f"({binned['datetime'].min()} .. {binned['datetime'].max()})")

    geomag = pd.read_csv(IN_GEOMAG, parse_dates=["datetime"])
    geomag["datetime"] = geomag["datetime"].dt.tz_localize(None) if geomag["datetime"].dt.tz else geomag["datetime"]
    geomag = geomag[["datetime", "Kp"]].rename(columns={"Kp": "kp"})

    merged = pd.merge(binned, geomag, on="datetime", how="left")
    merged["storm_kp6"] = merged["kp"] >= KP_STORM_THRESHOLD
    merged = merged.sort_values("datetime").reset_index(drop=True)

    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_TS, index=False)
    print(f"wrote {OUT_TS} ({len(merged)} rows)")

    intervals = storm_intervals(merged)
    storms_df = pd.DataFrame(intervals, columns=["start", "end"])
    storms_df.to_csv(OUT_STORMS, index=False)
    print(f"wrote {OUT_STORMS} ({len(storms_df)} Kp>=6 storm intervals)")
    print(storms_df.to_string(index=False))

    # 参考: 全期間の減衰率 (端点の3hビン平均の単純差分, 目視確認用)
    e0 = merged["E_J2_mean_Jkg"].iloc[:8].mean()   # 最初の1日
    e1 = merged["E_J2_mean_Jkg"].iloc[-8:].mean()  # 最後の1日
    days = (merged["datetime"].iloc[-1] - merged["datetime"].iloc[0]).total_seconds() / 86400.0
    print(f"\nfirst-day vs last-day mean E_J2: {e0:.1f} -> {e1:.1f} J/kg over {days:.1f} days "
          f"(mean rate {(e1 - e0) / days:.1f} J/kg/day, monotonic decay expected)")


if __name__ == "__main__":
    main()
