"""GOES-18 EUVS L2 (avg1m) 太陽EUV/FUV各波長データを3時間ビン平均に整形する。

対応TODO: #1 (軌道減衰指標の再検討 — EUV遅延相関の再計算に使う入力データ),
          #4 (Savitzky-Golayフィルタのアーティファクト検証),
          #8 (地磁気活動の定量的評価)

入力:
  data/external/goes_euv/{YYMM}_csv_combined.csv   (YYMM = 2404..2411, 8ファイル)
  GOES-18 EUVS L2 avg1m (1分値)。列: Time, AU_Factor, MgII_Index, MgII_Flag,
    irr_256, irr_284, irr_304, irr_1175, irr_1216, irr_1335, irr_1405,
    各 irr_*_flag_flag (0=正常)。Time はUTC。

処理:
  1. data/external/goes_euv/ の8ファイルの存在を確認する。
  2. 8ファイルを結合し、Time昇順にソート・重複時刻を除去。
  3. QC: 各波長 irr_XXX は irr_XXX_flag_flag == 0 の行のみ有効値として残す
     (それ以外は NaN)。MgII_Index は MgII_Flag == 0 の行のみ有効。
     フィルタは列ごとに独立に適用する (ある時刻で256nmが無効でも304nmは
     有効なら304nmの値は使う)。
  4. 3時間ビン (pandas resample("3h"), 既定の label="left", closed="left" =
     ビン開始時刻がラベル。dEdt_3h.csv・geomag_3h.csvと同じ00,03,06,...UTC
     グリッド) で各列の平均と有効サンプル数を計算する。
     **dEdt_3h.csv と異なり、データが皆無のビンも NaN として残し、
     ビングリッドを連続に保つ** (10_euv_lag_correlation.py でラグ相関計算時に
     時刻ベースで EUV(t-L) を直接インデックス参照できるようにするため)。

出力:
  data/processed/euv_3h.csv
    列: datetime, irr_256, irr_284, irr_304, irr_1175, irr_1216, irr_1335,
        irr_1405, MgII_Index,
        n_samples_irr_256, n_samples_irr_284, n_samples_irr_304,
        n_samples_irr_1175, n_samples_irr_1216, n_samples_irr_1335,
        n_samples_irr_1405, n_samples_MgII_Index

実行:
  .venv/Scripts/python src/09_prepare_euv.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_OUT_DIR = ROOT / "data" / "external" / "goes_euv"
OUT_PROCESSED = ROOT / "data" / "processed" / "euv_3h.csv"

YYMM_LIST = ["2404", "2405", "2406", "2407", "2408", "2409", "2410", "2411"]
WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
IRR_COLS = [f"irr_{wl}" for wl in WAVELENGTHS]
VALUE_COLS = IRR_COLS + ["MgII_Index"]
BIN = "3h"


def raw_files() -> list[Path]:
    files = [RAW_OUT_DIR / f"{yymm}_csv_combined.csv" for yymm in YYMM_LIST]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"GOES-18 EUVS files missing: {missing}")
    return files


def load_and_qc(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        d = pd.read_csv(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # NOTE: 2407/2408/2409 の元ファイルは "2024/6/28 0:00" 形式 (padding込みで
    # 隣接月に数日食い込む), 他は "2024-04-01 00:00:00" 形式。format="mixed" で
    # 両方を吸収する。重複時刻 (padding区間) は値が一致することを確認済み
    # (同一機器の生データなので四捨五入差以外は同値) なので単純に先勝ちで dedupe する。
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df.sort_values("Time").drop_duplicates("Time").set_index("Time")

    # QC: 各波長は自身の flag_flag==0 のみ有効。MgIIはMgII_Flag==0のみ有効。
    for wl in WAVELENGTHS:
        col = f"irr_{wl}"
        flag_col = f"{col}_flag_flag"
        bad = df[flag_col] != 0
        df.loc[bad, col] = float("nan")
        print(f"  {col}: {bad.sum()}/{len(df)} rows flagged (flag_flag!=0) -> NaN")

    bad_mgii = df["MgII_Flag"] != 0
    df.loc[bad_mgii, "MgII_Index"] = float("nan")
    print(f"  MgII_Index: {bad_mgii.sum()}/{len(df)} rows flagged (MgII_Flag!=0) -> NaN")

    return df[VALUE_COLS]


def bin_3h(df: pd.DataFrame) -> pd.DataFrame:
    agg = {c: ["mean", "count"] for c in VALUE_COLS}
    binned = df.resample(BIN).agg(agg)
    binned.columns = [
        (col if stat == "mean" else f"n_samples_{col}")
        for col, stat in binned.columns
    ]
    binned = binned.reset_index().rename(columns={"Time": "datetime"})
    return binned[
        ["datetime"] + VALUE_COLS + [f"n_samples_{c}" for c in VALUE_COLS]
    ]


def main() -> None:
    files = raw_files()

    print("\n=== QC (flag_flag == 0 のみ採用) ===")
    df = load_and_qc(files)
    print(f"\ncombined 1-min rows: {len(df)} "
          f"({df.index.min()} .. {df.index.max()})")

    binned = bin_3h(df)

    OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    binned.to_csv(OUT_PROCESSED, index=False)
    print(f"\nwrote {OUT_PROCESSED} ({len(binned)} rows, "
          f"{binned['datetime'].min()} .. {binned['datetime'].max()})")

    print("\n=== summary: valid-bin fraction per column ===")
    for c in VALUE_COLS:
        n_valid = binned[c].notna().sum()
        print(f"  {c:12s}: {n_valid}/{len(binned)} bins with data "
              f"({100*n_valid/len(binned):.1f}%)")


if __name__ == "__main__":
    main()
