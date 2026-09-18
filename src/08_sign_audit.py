"""新旧proxy相関の「符号反転」の原因診断 (SGフィルタを一切使わない分解)。

対応TODO: #1, #4

背景 (docs/2026-07-02_dEdt_proxy.md):
  新proxy da/dt (E_J2由来) と旧proxy d(Δh)/dt (Δh=TLE−GPS) の相関が、
  平滑化なしでは正 (Spearman +0.57)、SG平滑化 (N>=9) では負 (-0.4) になる。
  物理的期待は負。正相関の起源と、負相関がSGフィルタの人工物でないかを、
  SGフィルタ非依存の3診断で切り分ける。

入力:  data/processed/specific_energy.csv
       data/legacy_v1/altitude_diff_monthly_ext/*.csv
出力:  data/figure_data/sign_audit_lagdiff.csv      (診断1: ラグ差分相関 vs ラグ)
       data/figure_data/sign_audit_highpass_corr.csv (診断2: 高周波成分の相関行列)
       コンソールに診断3 (日次ブロック平均差分の相関) ほか要約

診断:
  1. ラグk差分相関: x_{t+k}-x_t と y_{t+k}-y_t の相関 (k=1..40ビン, 3h..5日)。
     SGなしで時間スケールを変える。符号がkとともに+から-へ遷移するなら、
     「負=低周波の実信号 / 正=高周波帯の現象」であり、SG平滑化は負相関を
     作ったのではなく高周波を除去して露出させただけと結論できる。
  2. 高周波残差 (24h centered rolling meanからの残差) の相関行列:
     {E_J2, GPS高度, TLE高度, Δh} → 正相関がGPS側/TLE側どちら由来か特定。
  3. 日次ブロック平均→単純1日差分 (重複なし・フィルタなし) の新旧相関。

実行:  .venv/Scripts/python src/08_sign_audit.py
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_NEW = ROOT / "data" / "processed" / "specific_energy.csv"
IN_OLD_DIR = ROOT / "data" / "legacy_v1" / "altitude_diff_monthly_ext"
OUT_LAG = ROOT / "data" / "figure_data" / "sign_audit_lagdiff.csv"
OUT_HP = ROOT / "data" / "figure_data" / "sign_audit_highpass_corr.csv"

MU = 3.986004418e14
BIN = "3h"

MONTH_CORE_RANGE = {
    "2404": ("2024-04-01", "2024-05-01"), "2405": ("2024-05-01", "2024-06-01"),
    "2406": ("2024-06-01", "2024-07-01"), "2407": ("2024-07-01", "2024-08-01"),
    "2408": ("2024-08-01", "2024-09-01"), "2409": ("2024-09-01", "2024-10-01"),
    "2410": ("2024-10-01", "2024-11-01"), "2411": ("2024-11-01", "2024-12-01"),
}


def load_merged_bins() -> pd.DataFrame:
    """新旧の元系列 (レベル) を共通3hグリッドで結合して返す。微分はしない。"""
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


def lag_diff_corr(merged: pd.DataFrame, max_lag_bins: int = 40) -> pd.DataFrame:
    """診断1: ラグk差分同士の相関。連続区間のみ使用 (欠測跨ぎの差分は除外)。"""
    # 3hグリッド上の整数位置に展開して欠測跨ぎを検出できるようにする
    t0 = merged.index[0]
    pos = ((merged.index - t0).total_seconds() / (3 * 3600)).astype(int)
    e = pd.Series(merged["E_J2"].to_numpy(), index=pos)
    dh = pd.Series(merged["altitude_diff"].to_numpy(), index=pos)
    full = pd.RangeIndex(pos.min(), pos.max() + 1)
    e = e.reindex(full)
    dh = dh.reindex(full)

    rows = []
    for k in range(1, max_lag_bins + 1):
        de = e.diff(k)
        ddh = dh.diff(k)
        ok = de.notna() & ddh.notna()
        if ok.sum() < 50:
            continue
        rp, _ = pearsonr(de[ok], ddh[ok])
        rs, _ = spearmanr(de[ok], ddh[ok])
        rows.append({"lag_bins": k, "lag_hours": 3 * k, "n": int(ok.sum()),
                     "pearson_r": rp, "spearman_r": rs})
    return pd.DataFrame(rows)


def highpass_corr(merged: pd.DataFrame) -> pd.DataFrame:
    """診断2: 24h centered rolling meanからの残差同士の相関行列。"""
    cols = {"E_J2": merged["E_J2"],
            "gps_alt": merged["gps_average_altitude"],
            "tle_alt": merged["tle_average_altitude"],
            "delta_h": merged["altitude_diff"]}
    hp = {}
    for name, s in cols.items():
        s = s.asfreq("3h") if hasattr(s, "asfreq") else s
        roll = s.rolling(9, center=True, min_periods=5).mean()
        hp[name] = s - roll
    hp = pd.DataFrame(hp).dropna()
    corr = hp.corr(method="spearman")
    return corr


def daily_block_diff(merged: pd.DataFrame) -> None:
    """診断3: 日次ブロック平均 → 単純1日差分 (フィルタ非依存) の新旧相関。"""
    daily = merged.resample("1D").agg(
        E_J2=("E_J2", "mean"), a_eq=("a_eq", "mean"),
        delta_h=("altitude_diff", "mean"), n=("E_J2", "count"))
    daily = daily[daily["n"] >= 4]  # 1日の半分以上ビンがある日のみ
    # 連続日のみ差分
    dt_days = daily.index.to_series().diff().dt.days
    dE = daily["E_J2"].diff()
    dDh = daily["delta_h"].diff()
    ok = (dt_days == 1) & dE.notna() & dDh.notna()
    a2 = daily["a_eq"].mean() ** 2
    dadt = (2 * a2 / MU) * dE[ok]          # m/day相当 (定数倍なので相関に影響なし)
    ddhdt = dDh[ok] * 1000.0               # km→m /day
    rp, pp = pearsonr(dadt, ddhdt)
    rs, ps = spearmanr(dadt, ddhdt)
    print("\n=== 診断3: 日次ブロック平均の単純1日差分 (フィルタ完全非依存) ===")
    print(f"n = {ok.sum()} 日")
    print(f"Pearson  r = {rp:+.4f} (p = {pp:.3g})")
    print(f"Spearman r = {rs:+.4f} (p = {ps:.3g})")
    print("(負なら『物理的期待どおりの負相関はフィルタ非依存で存在する』ことの確証)")


def main() -> None:
    merged = load_merged_bins()
    print(f"common 3h bins: {len(merged)} "
          f"({merged.index.min()} .. {merged.index.max()})")

    lag = lag_diff_corr(merged)
    OUT_LAG.parent.mkdir(parents=True, exist_ok=True)
    lag.to_csv(OUT_LAG, index=False)
    print(f"\n=== 診断1: ラグ差分相関 (SG不使用) → {OUT_LAG} ===")
    show = lag[lag["lag_bins"].isin([1, 2, 4, 8, 12, 16, 24, 32, 40])]
    print(show.to_string(index=False))

    corr = highpass_corr(merged)
    corr.to_csv(OUT_HP)
    print(f"\n=== 診断2: 高周波残差 (24h rolling mean除去) のSpearman相関行列 → {OUT_HP} ===")
    print(corr.round(3).to_string())

    daily_block_diff(merged)


if __name__ == "__main__":
    main()
