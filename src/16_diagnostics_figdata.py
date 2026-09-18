"""診断・感度解析パネルのうち、図データCSVがまだ存在しない2件を生成する。

対応TODO: #1, #4, #8, #9, #20 (診断・感度解析の可視化刷新に必要な図データ整備)

背景:
  revision_analysis の「診断・感度解析」領域 (符号監査・地磁気・プリホワイトニング・
  ブートストラップCI・平滑化対称化・ビン幅) はほとんどの解析で既に
  data/figure_data/*.csv が揃っているが、以下2件だけは図に必要なCSVが存在しない:
    (1) sign_audit の診断3 (日次ブロック平均→単純1日差分) は
        src/08_sign_audit.py の daily_block_diff() 内でコンソール出力のみ
        (Spearman/Pearson要約値) を行い、散布図に必要な日別ペア (n=200点) を
        保存していない。
    (2) geomag の月別Kp>=3/Kp>=4割合、および全期間3hグリッドの静穏マスク
        (直近48h max Kp<4) は data/processed/geomag_3h.csv から都度計算されて
        おり、figure_data化されていない。

本スクリプトはこの2件だけを生成する。src/08_sign_audit.py 自体は変更禁止のため、
(1) の再現に必要なロジック (日次ブロック平均→連続日のみ1日差分→新旧proxy換算)
を本スクリプト内に独立に再実装する (ロジックはsrc/08と同一、出力のみ追加)。
(2) は src/05_geomagnetic_indices.py の出力 (geomag_3h.csv) をそのまま集計する
だけで、10_euv_lag_correlation.py と同じ「直近48h trailing max」の定義
(KP_ROLL_BINS=16) を再利用する。

入力:
  data/processed/specific_energy.csv                 (02_specific_energy.py)
  data/legacy_v1/altitude_diff_monthly_ext/*.csv    (旧proxyアーカイブ, MT_codeから
                                                        コピー済み。08と同じ入力)
  data/processed/geomag_3h.csv                        (05_geomagnetic_indices.py)

出力:
  data/figure_data/sign_audit_daily_diff.csv
      columns: date, dadt_m_per_day, ddhdt_m_per_day
      (08_sign_audit.py の diag3 と同一定義: 日次ブロック平均 [n>=4 bin/day] の
       連続日のみの1日差分。dadt = (2*a_eq_mean^2/mu) * dE_J2 [m/day相当],
       ddhdt = d(altitude_diff) * 1000 [km/day -> m/day])
  data/figure_data/geomag_quiet_mask_3h.csv
      columns: datetime, Kp, kp_max48h, quiet_kp4
      (解析対象期間 2024-04-01〜2024-11-30 のみ。kp_max48hは
       10_euv_lag_correlation.py と同じ trailing 48h [3h x 16bin] max)
  data/figure_data/geomag_monthly_summary.csv
      columns: month, n, frac_kp_ge3, frac_kp_ge4

実行:
  .venv/Scripts/python src/16_diagnostics_figdata.py
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]

# --- sign_audit daily diff (diag3 re-implementation, matches src/08 exactly) ---
IN_NEW = ROOT / "data" / "processed" / "specific_energy.csv"
IN_OLD_DIR = ROOT / "data" / "legacy_v1" / "altitude_diff_monthly_ext"
OUT_DAILY_DIFF = ROOT / "data" / "figure_data" / "sign_audit_daily_diff.csv"

MU = 3.986004418e14
BIN = "3h"

MONTH_CORE_RANGE = {
    "2404": ("2024-04-01", "2024-05-01"), "2405": ("2024-05-01", "2024-06-01"),
    "2406": ("2024-06-01", "2024-07-01"), "2407": ("2024-07-01", "2024-08-01"),
    "2408": ("2024-08-01", "2024-09-01"), "2409": ("2024-09-01", "2024-10-01"),
    "2410": ("2024-10-01", "2024-11-01"), "2411": ("2024-11-01", "2024-12-01"),
}

# --- geomag figure data ---
IN_GEOMAG = ROOT / "data" / "processed" / "geomag_3h.csv"
OUT_GEOMAG_3H = ROOT / "data" / "figure_data" / "geomag_quiet_mask_3h.csv"
OUT_GEOMAG_MONTHLY = ROOT / "data" / "figure_data" / "geomag_monthly_summary.csv"

KP_ROLL_BINS = 16  # 3h x 16 = 48h trailing window (10_euv_lag_correlation.py と同じ)
PERIOD_START = pd.Timestamp("2024-04-01")
PERIOD_END = pd.Timestamp("2024-12-01")  # exclusive


# ===================================================================
# (1) sign_audit: 診断3 の日別ペアをCSVに保存 (src/08_sign_audit.py と同一ロジック)
# ===================================================================

def load_merged_bins() -> pd.DataFrame:
    """新旧の元系列 (レベル) を共通3hグリッドで結合して返す。微分はしない。
    (src/08_sign_audit.py::load_merged_bins の再実装, 完全同一ロジック)
    """
    new = pd.read_csv(IN_NEW, parse_dates=["datetime"]).set_index("datetime")
    new_b = new.resample(BIN).agg(
        E_J2=("E_J2_Jkg", "mean"), a_eq=("a_equiv_m", "mean"),
        n=("E_J2_Jkg", "count"))
    new_b = new_b[new_b["n"] > 0]

    frames = []
    for f in sorted(glob.glob(str(IN_OLD_DIR / "*_monthly_ext.csv"))):
        yymm = Path(f).stem.split("_")[-3]
        start, end = MONTH_CORE_RANGE[yymm]
        d = pd.read_csv(f)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        d = d.dropna(subset=["timestamp", "altitude_diff"])
        m = (d["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (
            d["timestamp"] < pd.Timestamp(end, tz="UTC"))
        frames.append(d.loc[m, ["timestamp", "gps_average_altitude",
                                "tle_average_altitude", "altitude_diff"]])
    old = (pd.concat(frames).drop_duplicates("timestamp").set_index("timestamp")
           .sort_index())
    old_b = old.resample(BIN).mean().dropna()

    merged = new_b.join(old_b, how="inner").dropna()
    return merged


def daily_block_diff_table(merged: pd.DataFrame) -> pd.DataFrame:
    """診断3: 日次ブロック平均 -> 単純1日差分 (フィルタ非依存) の日別ペアを返す。

    src/08_sign_audit.py::daily_block_diff と同一の日選定・同一の物理量換算だが、
    要約統計だけでなく日別ペア (date, dadt, ddhdt) をそのままDataFrameで返す点が
    異なる (08はコンソール要約のみでCSVに保存しない)。
    """
    daily = merged.resample("1D").agg(
        E_J2=("E_J2", "mean"), a_eq=("a_eq", "mean"),
        delta_h=("altitude_diff", "mean"), n=("E_J2", "count"))
    daily = daily[daily["n"] >= 4]  # 1日の半分以上ビンがある日のみ
    dt_days = daily.index.to_series().diff().dt.days
    dE = daily["E_J2"].diff()
    dDh = daily["delta_h"].diff()
    ok = (dt_days == 1) & dE.notna() & dDh.notna()

    a2 = daily["a_eq"].mean() ** 2
    dadt = (2 * a2 / MU) * dE[ok]     # m/day相当 (新proxy da/dt)
    ddhdt = dDh[ok] * 1000.0          # km/day -> m/day (旧proxy d(Delta h)/dt)

    out = pd.DataFrame({
        "date": daily.index[ok],
        "dadt_m_per_day": dadt.to_numpy(),
        "ddhdt_m_per_day": ddhdt.to_numpy(),
    })
    return out.reset_index(drop=True)


def build_sign_audit_daily_diff() -> None:
    merged = load_merged_bins()
    table = daily_block_diff_table(merged)
    OUT_DAILY_DIFF.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DAILY_DIFF, index=False)

    rp, pp = pearsonr(table["dadt_m_per_day"], table["ddhdt_m_per_day"])
    rs, ps = spearmanr(table["dadt_m_per_day"], table["ddhdt_m_per_day"])
    print(f"wrote {OUT_DAILY_DIFF} ({len(table)} rows)")
    print(f"  check (must match src/08_sign_audit.py console diag3): "
          f"n={len(table)}  Pearson r={rp:+.4f} (p={pp:.3g})  "
          f"Spearman r={rs:+.4f} (p={ps:.3g})")


# ===================================================================
# (2) geomag: 静穏マスク付き3hグリッド + 月別Kp>=3/Kp>=4割合
# ===================================================================

def build_geomag_figdata() -> None:
    df = pd.read_csv(IN_GEOMAG, parse_dates=["datetime"]).sort_values("datetime")
    df = df.reset_index(drop=True)
    step = df["datetime"].diff().dropna().unique()
    assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
        f"geomag_3h.csv is not a continuous 3h grid: {step}"

    # kp_max48h は解析全域 (パディング込み) で計算してから解析期間で切る
    # (10_euv_lag_correlation.py::load_geomag_kp_max48h と同じ定義・同じ計算順)
    df["kp_max48h"] = df["Kp"].rolling(KP_ROLL_BINS, min_periods=1).max()

    full = df[(df["datetime"] >= PERIOD_START) & (df["datetime"] < PERIOD_END)].copy()
    full["quiet_kp4"] = full["kp_max48h"] < 4.0
    out3h = full[["datetime", "Kp", "kp_max48h", "quiet_kp4"]].reset_index(drop=True)
    OUT_GEOMAG_3H.parent.mkdir(parents=True, exist_ok=True)
    out3h.to_csv(OUT_GEOMAG_3H, index=False)
    print(f"wrote {OUT_GEOMAG_3H} ({len(out3h)} rows, "
          f"{out3h['datetime'].min()} .. {out3h['datetime'].max()})")

    # 月別 Kp>=3 / Kp>=4 の3hビン割合 (瞬時Kp基準, kp_max48hではない)
    monthly = full.copy()
    monthly["month"] = monthly["datetime"].dt.to_period("M").astype(str)
    summary = monthly.groupby("month").apply(
        lambda g: pd.Series({
            "n": len(g),
            "frac_kp_ge3": float((g["Kp"] >= 3.0).mean()),
            "frac_kp_ge4": float((g["Kp"] >= 4.0).mean()),
        }), include_groups=False
    ).reset_index()
    summary["n"] = summary["n"].astype(int)
    summary.to_csv(OUT_GEOMAG_MONTHLY, index=False)
    print(f"wrote {OUT_GEOMAG_MONTHLY} ({len(summary)} rows)")
    print(summary.to_string(index=False))


def main() -> None:
    print("=== (1) sign_audit daily-block-diff scatter data ===")
    build_sign_audit_daily_diff()
    print("\n=== (2) geomag overview figure data ===")
    build_geomag_figdata()


if __name__ == "__main__":
    main()
