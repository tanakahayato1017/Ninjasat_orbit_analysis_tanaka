"""GPSテレメトリQC後データの品質全体像を集計する (図1: gps_qc_overview)。

対応TODO: #1 の前段可視化 (基盤パイプラインの土台になるQCデータの俯瞰)

背景:
  docs/2026-07-02_specific_energy_method.md の解析はすべて
  data/processed/gps_qc.csv (01_prepare_gps.py の出力, QC通過29,267点) を
  出発点にする。このQC後データが「いつ・どれだけ・どの品質で」取れているかを
  一枚で示す図のための集計をここで行う (描画は plots/plot_gps_qc_overview.py)。

入力:  data/processed/gps_qc.csv

出力:
  data/figure_data/gps_qc_overview_daily.csv
    列: date, n_samples  (日ごとのQC通過サンプル数)
  data/figure_data/gps_qc_overview_gaps.csv
    列: gap_start, gap_end, duration_hours  (連続サンプル間隔が6h超のギャップ一覧。
    6hしきい値は03_dEdt_proxy.pyの欠測跨ぎ判定と同じ基準)
  data/figure_data/gps_qc_overview_sats_hist.csv
    列: bin_left, bin_right, count  (gpsNumOfSatsTrackedのヒストグラム, 整数ビン幅1)
  data/figure_data/gps_qc_overview_hdop_hist.csv
    列: bin_left, bin_right, count  (gpsHDilutionOfPosのヒストグラム, 0.05刻み)

実行: .venv/Scripts/python src/17_qc_overview_figdata.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "gps_qc.csv"
OUT_DAILY = ROOT / "data" / "figure_data" / "gps_qc_overview_daily.csv"
OUT_GAPS = ROOT / "data" / "figure_data" / "gps_qc_overview_gaps.csv"
OUT_SATS = ROOT / "data" / "figure_data" / "gps_qc_overview_sats_hist.csv"
OUT_HDOP = ROOT / "data" / "figure_data" / "gps_qc_overview_hdop_hist.csv"

GAP_THRESHOLD_HOURS = 6.0   # 03_dEdt_proxy.py の欠測跨ぎ判定 (>6h) と同じ基準


def main() -> None:
    df = pd.read_csv(IN, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # --- 日毎サンプル数 ---
    daily = (
        df.assign(date=df["datetime"].dt.floor("D"))
        .groupby("date")
        .size()
        .rename("n_samples")
        .reset_index()
    )
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DAILY, index=False)
    print(f"wrote {OUT_DAILY} ({len(daily)} days, "
          f"{daily['date'].min().date()} .. {daily['date'].max().date()})")
    print(f"  n_samples/day: median={daily['n_samples'].median():.0f}, "
          f"min={daily['n_samples'].min()}, max={daily['n_samples'].max()}")

    # --- 欠測ギャップ (連続サンプル間隔 > 6h) ---
    dt = df["datetime"].diff()
    gap_mask = dt > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)
    gap_idx = df.index[gap_mask]
    gaps = pd.DataFrame({
        "gap_start": df.loc[gap_idx - 1, "datetime"].to_numpy(),
        "gap_end": df.loc[gap_idx, "datetime"].to_numpy(),
    })
    gaps["duration_hours"] = (
        (pd.to_datetime(gaps["gap_end"]) - pd.to_datetime(gaps["gap_start"]))
        .dt.total_seconds() / 3600.0
    )
    gaps = gaps.sort_values("duration_hours", ascending=False).reset_index(drop=True)
    gaps.to_csv(OUT_GAPS, index=False)
    print(f"\nwrote {OUT_GAPS} ({len(gaps)} gaps > {GAP_THRESHOLD_HOURS}h)")
    print(gaps.to_string(index=False))

    # --- 追尾衛星数ヒストグラム (整数ビン) ---
    sats = df["gpsNumOfSatsTracked"].dropna().to_numpy()
    lo, hi = int(np.floor(sats.min())), int(np.ceil(sats.max())) + 1
    bin_edges = np.arange(lo, hi + 1, 1)
    counts, edges = np.histogram(sats, bins=bin_edges)
    sats_hist = pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:], "count": counts,
    })
    sats_hist.to_csv(OUT_SATS, index=False)
    print(f"\nwrote {OUT_SATS} (n={len(sats)}, median sats={np.median(sats):.0f}, "
          f"p10={np.percentile(sats, 10):.0f}, p90={np.percentile(sats, 90):.0f})")

    # --- HDOPヒストグラム (0.05刻み) ---
    if "gpsHDilutionOfPos" not in df.columns:
        print("NOTE: gpsHDilutionOfPos not in the public GNSS file; HDOP histogram skipped.")
        return
    hdop = df["gpsHDilutionOfPos"].dropna().to_numpy()
    bin_edges = np.arange(0.0, np.ceil(hdop.max() * 20) / 20 + 0.05, 0.05)
    counts, edges = np.histogram(hdop, bins=bin_edges)
    hdop_hist = pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:], "count": counts,
    })
    hdop_hist.to_csv(OUT_HDOP, index=False)
    print(f"wrote {OUT_HDOP} (n={len(hdop)}, median HDOP={np.median(hdop):.2f}, "
          f"p10={np.percentile(hdop, 10):.2f}, p90={np.percentile(hdop, 90):.2f})")


if __name__ == "__main__":
    main()
