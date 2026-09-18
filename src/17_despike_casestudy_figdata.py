"""despikeケーススタディ用の図データ抽出 (図3: despike_casestudy)。

対応TODO: #6 (不確かさの定量化)

背景:
  docs/2026-07-02_spike_diagnosis.md / docs/2026-07-02_pipeline_remediation.md で
  実施したサンプルレベルdespike (02_specific_energy.py, 13サンプル中心移動中央値
  からの乖離が全期間の大域MADの10倍を超えるサンプルを除去) の効果を、
  (a) 実際に除去された典型例 (2024-10-25、劣化GNSS fix由来の単発+137 kJ/kg外れ値)
  (b) 除去されなかった実信号の例 (2024-05-10〜12、Gannon磁気嵐、除去0件)
  の2ケースで示す図のためのデータを、despike前のバックアップ
  (data/processed/backup_pre_remediation/specific_energy.csv) から抽出する。

  02_specific_energy.py の despike_e_j2() と全く同じロジック (窓13・
  min_periods7の中心移動中央値、大域MAD閾値10倍) をこのスクリプト内で
  再実装し、despike前の全期間データに適用する。02_specific_energy.pyは
  変更禁止のため、ここで独立に再現する (入力が同一なので結果は本番の
  despikeと一致するはずであり、これは実装の相互検証にもなる)。

入力:  data/processed/backup_pre_remediation/specific_energy.csv (despike前, 全期間)

出力:  data/figure_data/despike_casestudy.csv
  列: datetime, panel, E_J2_Jkg, roll_median_Jkg, threshold_lo_Jkg,
      threshold_hi_Jkg, removed
  panel = "oct25" (2024-10-25 00:00-09:00 UTC) または
          "gannon" (2024-05-10 00:00 - 2024-05-12 00:00 UTC, Gannon storm)

実行: .venv/Scripts/python src/17_despike_casestudy_figdata.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "backup_pre_remediation" / "specific_energy.csv"
OUT = ROOT / "data" / "figure_data" / "despike_casestudy.csv"

# 02_specific_energy.py の despike_e_j2() と同一のパラメータ
DESPIKE_WINDOW = 13
DESPIKE_MIN_PERIODS = 7
DESPIKE_MAD_MULTIPLIER = 10.0

PANEL_WINDOWS = {
    "oct25": (pd.Timestamp("2024-10-25 00:00", tz="UTC"), pd.Timestamp("2024-10-25 09:00", tz="UTC")),
    "gannon": (pd.Timestamp("2024-05-10 00:00", tz="UTC"), pd.Timestamp("2024-05-12 00:00", tz="UTC")),
}


def main() -> None:
    df = pd.read_csv(IN, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    col = "E_J2_Jkg"
    roll_median = df[col].rolling(
        window=DESPIKE_WINDOW, center=True, min_periods=DESPIKE_MIN_PERIODS
    ).median()
    dev = (df[col] - roll_median).abs()
    mad = dev.median()
    threshold = DESPIKE_MAD_MULTIPLIER * mad
    removed = dev > threshold

    print(f"reproduced despike: global MAD={mad:.2f} J/kg, threshold={threshold:.2f} J/kg, "
          f"removed {int(removed.sum())}/{len(df)} samples "
          "(should match pipeline_remediation.md: MAD=293.96, threshold=2939.6, removed=42)")

    df["roll_median_Jkg"] = roll_median
    df["threshold_lo_Jkg"] = roll_median - threshold
    df["threshold_hi_Jkg"] = roll_median + threshold
    df["removed"] = removed

    chunks = []
    for panel, (t0, t1) in PANEL_WINDOWS.items():
        mask = (df["datetime"] >= t0) & (df["datetime"] < t1)
        sub = df.loc[mask, ["datetime", col, "roll_median_Jkg", "threshold_lo_Jkg",
                             "threshold_hi_Jkg", "removed"]].copy()
        sub["panel"] = panel
        n_removed = int(sub["removed"].sum())
        print(f"  panel={panel:8s} window=[{t0}, {t1})  n={len(sub)}  removed={n_removed}")
        chunks.append(sub)

    out = pd.concat(chunks, ignore_index=True)
    out = out[["datetime", "panel", col, "roll_median_Jkg", "threshold_lo_Jkg",
               "threshold_hi_Jkg", "removed"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
