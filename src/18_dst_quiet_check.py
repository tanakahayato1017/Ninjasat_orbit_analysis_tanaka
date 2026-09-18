"""Dstベースの静穏期マスクによるEUV遅延相関クロスチェック。

対応TODO: #8 (地磁気活動の定量的評価 — R2 Minor#1「Kp/Ap/Dstが高い期間を除いても
          遅延が不変か示せ」のうち、Kpのみで未対応だったDstを補完する)

背景:
  10_euv_lag_correlation.py は「直近48hの最大Kp」で静穏期を定義したが、R2は
  Kp/Ap/Dstを列挙しており、低緯度リングカレント現象への感度はDstの方が高い。
  本スクリプトはDstベースの静穏期マスクでも同じ結論（地磁気活動を制御すると
  長ラグがEUV/FUV帯30-36hに収斂する、という10/quiet_time_lag_methodの結論）
  になるかを確認する。
  （docs/2026-07-02_quiet_time_lag_method.md 「限界と注意」参照）

入力:
  data/external/omni2/omni2_2024.dat  (NASA SPDF OMNI2 低分解能1h値、事前にcurlで取得)
  data/external/omni2/omni2.text      (OMNI2フォーマット定義。word 41 = Dst Index,
                                        I6, fill=99999, nT, Kyoto発表値)
  data/processed/dEdt_3h.csv          (03_dEdt_proxy.py の出力、平滑化なし)
  data/processed/euv_3h.csv           (09_prepare_euv.py の出力、連続3hグリッド)
  data/processed/geomag_3h.csv        (05_geomagnetic_indices.py の出力、
                                        図3用のKp>=6嵐イベント抽出にのみ使用)

手法:
  1. OMNI2 1h値ファイルをフォーマット定義(word 41 = DST Index, I6, fill=99999)に
     従って解析し、2024-03-25〜2024-12-05 UTCの Dst_nT を抽出。
     data/processed/dst_hourly.csv (datetime, Dst_nT) に保存。
     検証: 2024-05-11 (Gannon storm) のDst最小値が既知の-412nT前後と整合するか確認。
  2. Dstを3hグリッド (00,03,...UTC, dEdt_3h.csv/euv_3h.csv/geomag_3h.csvと同じ
     ビン境界) に変換: 各3hビン内の最小Dst (最も擾乱が強い瞬間を代表させる)。
       Dst_min_3h(bin) = min_{s in bin} Dst(s)
  3. 直近48時間 (trailing, 3hビン16個, t-45h〜t) の最小Dst:
       Dst_min48h(t) = min_{t-45h<=s<=t} Dst_min_3h(s)
     （10_euv_lag_correlation.py の Kp^max_48h と同じtrailing窓構造。Dstは
     擾乱が強いほど負に大きくなるため min を使う点のみがKpのmaxと異なる）
  4. マスク2種 (proxy側のビンtにのみ適用、EUV側は地磁気の影響を受けないため無適用。
     10_euv_lag_correlation.py と同じ設計判断):
       quiet_dst30: Dst_min48h(t) > -30 nT
       quiet_dst50: Dst_min48h(t) > -50 nT
  5. proxy = -dEdt_Jkg_per_day。外れ値除去は10/14と同じ modified z-score
     (Iglewicz & Hoaglin) |z_mod| > 8.0。SG平滑化はwindow=13点(=39h,polyorder=1)
     のみ (10_euv_lag_correlation.py の主要結果表と同じ窓)。期間はfull
     (2024-04-01〜2024-11-30) のみ。
  6. ラグ相関: L = 0, 3, ..., 120 [h] について
       spearman_r = Spearman corr( EUV(t-L), proxy(t) )
     10_euv_lag_correlation.py と同じ整数グリッド参照 (線形補間なし)。
  7. ピーク: 各 (mask, wavelength) につき lag_hours>=6 の範囲でspearman_rが
     最大となるラグ (10と同じ定義、lag<6hの自明な同時相関を除外)。
  8. Kpベース (data/figure_data/euv_lag_peaks.csv の window=13, period=full,
     geomag in {all, quiet_kp4, quiet_kp3}) とDstベースのピークラグを並べた
     比較表を作り、結論が変わるかを判定してコンソールに出力する。
  9. 図3用の補助データ: quiet_dst50が成立する3h区間のリスト、およびKp>=6の
     磁気嵐イベント(開始/終了/ピークKp/ピーク時刻)を抽出して保存する
     (plot_geomag_overview.py と同じアルゴリズムをこのスクリプト内で独立に
     再実装。既存スクリプトは変更しない)。

出力:
  data/processed/dst_hourly.csv
      columns: datetime, Dst_nT  (1h値、2024-03-25〜2024-12-05 UTC)
  data/figure_data/dst_quiet_check.csv
      columns: mask, wavelength, lag_hours, spearman_r, n
  data/figure_data/dst_quiet_peaks.csv
      columns: mask, wavelength, peak_lag_hours, peak_r, n
  data/figure_data/dst_kp_peak_comparison.csv
      columns: wavelength, all, quiet_kp4, quiet_kp3, quiet_dst50, quiet_dst30
      (各セルはピークラグ[h]。全てwindow=13, period=full)
  data/figure_data/dst_quiet_mask_3h.csv
      columns: datetime, Dst_min_3h, dst_min48h, quiet_dst30, quiet_dst50
      (解析期間 full=2024-04-01〜2024-11-30 の3hグリッド。図3上段の背景帯用)
  data/figure_data/dst_storm_events_kp6.csv
      columns: start, end, peak_kp, peak_time  (Kp>=6の連続区間。図3上段の注記用)

実行:
  .venv/Scripts/python src/18_dst_quiet_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]

IN_OMNI2 = ROOT / "data" / "external" / "omni2" / "omni2_2024.dat"
IN_DEDT = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_EUV = ROOT / "data" / "processed" / "euv_3h.csv"
IN_GEOMAG = ROOT / "data" / "processed" / "geomag_3h.csv"
IN_EUV_LAG_PEAKS = ROOT / "data" / "figure_data" / "euv_lag_peaks.csv"

OUT_DST_HOURLY = ROOT / "data" / "processed" / "dst_hourly.csv"
OUT_CORR = ROOT / "data" / "figure_data" / "dst_quiet_check.csv"
OUT_PEAKS = ROOT / "data" / "figure_data" / "dst_quiet_peaks.csv"
OUT_COMPARISON = ROOT / "data" / "figure_data" / "dst_kp_peak_comparison.csv"
OUT_MASK_3H = ROOT / "data" / "figure_data" / "dst_quiet_mask_3h.csv"
OUT_STORM_EVENTS = ROOT / "data" / "figure_data" / "dst_storm_events_kp6.csv"

# Extraction period for the hourly Dst product (task #1)
DST_START = pd.Timestamp("2024-03-25 00:00:00")
DST_END = pd.Timestamp("2024-12-05 23:59:59")

# Analysis period for lag correlation (same as 10_euv_lag_correlation.py "full")
PERIOD_START = pd.Timestamp("2024-04-01")
PERIOD_END = pd.Timestamp("2024-12-01")  # exclusive

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
LAGS_H = list(range(0, 121, 3))             # 0,3,...,120 (41 lags)
PEAK_MIN_LAG_H = 6                          # lag<6h の自明な同時相関は除外

SG_WINDOW = 13                              # 10_euv_lag_correlation.py の主要窓
OUTLIER_THRESHOLD = 8.0                     # 07/10/14 と同じMAD基準
MIN_N = 10
ROLL_BINS_48H = 16                          # 3h x 16 = 48h trailing window

DST_MASKS = {
    "quiet_dst30": -30.0,
    "quiet_dst50": -50.0,
}

STORM_KP_THRESHOLD = 6.0
BIN_TD = pd.Timedelta("3h")

# OMNI2 hourly format (omni2.text): 55 whitespace-separated words per line.
# word 1 = Year, word 2 = Decimal Day (Jan 1 = 1), word 3 = Hour (0-23),
# word 41 = DST Index [nT], fill value 99999 (I6).
OMNI2_YEAR_COL = 0
OMNI2_DOY_COL = 1
OMNI2_HOUR_COL = 2
OMNI2_DST_COL = 40   # word 41, zero-based index 40
OMNI2_DST_FILL = 99999
OMNI2_NCOLS = 55


# =====================================================================
# Step 1: parse OMNI2 hourly Dst
# =====================================================================

def parse_omni2_dst() -> pd.DataFrame:
    raw = pd.read_csv(IN_OMNI2, sep=r"\s+", header=None)
    assert raw.shape[1] == OMNI2_NCOLS, (
        f"unexpected OMNI2 field count: {raw.shape[1]} (expected {OMNI2_NCOLS})"
    )

    year = raw[OMNI2_YEAR_COL].astype(int)
    doy = raw[OMNI2_DOY_COL].astype(int)
    hour = raw[OMNI2_HOUR_COL].astype(int)
    # Jan 1 = day 1 -> datetime = (Jan 1 00:00 of that year) + (doy-1) days + hour
    datetime_col = (
        pd.to_datetime(year.astype(str) + "-01-01")
        + pd.to_timedelta(doy - 1, unit="D")
        + pd.to_timedelta(hour, unit="h")
    )

    dst = raw[OMNI2_DST_COL].astype(float)
    n_fill = int((dst == OMNI2_DST_FILL).sum())
    dst = dst.replace(float(OMNI2_DST_FILL), np.nan)

    df = pd.DataFrame({"datetime": datetime_col, "Dst_nT": dst})
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"parsed OMNI2 hourly file: {len(df)} rows "
          f"({df['datetime'].min()} .. {df['datetime'].max()}), "
          f"{n_fill} fill-value (99999) Dst rows in full year")

    sel = df[(df["datetime"] >= DST_START) & (df["datetime"] <= DST_END)].copy()
    sel = sel.reset_index(drop=True)
    n_nan = int(sel["Dst_nT"].isna().sum())
    print(f"extracted {DST_START.date()} .. {DST_END.date()}: {len(sel)} hourly rows, "
          f"{n_nan} missing (NaN) Dst values")

    OUT_DST_HOURLY.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_DST_HOURLY, index=False)
    print(f"wrote {OUT_DST_HOURLY}")

    # --- Verification: Gannon storm (2024-05-11) minimum Dst ---
    gannon_day = sel[(sel["datetime"] >= "2024-05-11") & (sel["datetime"] < "2024-05-12")]
    if len(gannon_day):
        idx_min = gannon_day["Dst_nT"].idxmin()
        print(f"\n[verification] Gannon storm 2024-05-11 minimum Dst: "
              f"{gannon_day.loc[idx_min, 'Dst_nT']:.0f} nT "
              f"at {gannon_day.loc[idx_min, 'datetime']}")
        print("  known published minimum (Kyoto WDC provisional/final Dst): "
              "approx. -412 nT at 02 UT on 2024-05-11 -- OMNI2 value should be "
              "within a few nT of this (small differences from provisional vs. "
              "final Dst revision are expected).")
    else:
        print("\n[verification] WARNING: no 2024-05-11 rows found in extracted range")

    return df  # full-year (for 3h-grid construction with pre-period buffer)


# =====================================================================
# Step 2: 3h grid + trailing 48h min + quiet masks
# =====================================================================

def build_dst_3h(dst_hourly_full: pd.DataFrame) -> pd.DataFrame:
    """1h Dstを3hグリッド最小値に変換し、直近48h最小値と静穏マスクを付与する。

    dEdt_3h.csv/euv_3h.csv/geomag_3h.csvと同じビン境界 (00,03,...UTC,
    label=left, closed=left) を使う。トレイリング窓の立ち上がり期間を
    確保するため、解析期間より前 (DST_START〜) から連続3hグリッドを作る。
    """
    df = dst_hourly_full[
        (dst_hourly_full["datetime"] >= DST_START) &
        (dst_hourly_full["datetime"] <= DST_END)
    ].copy()
    df = df.set_index("datetime").sort_index()

    binned = df["Dst_nT"].resample("3h", label="left", closed="left").min()
    binned = binned.to_frame("Dst_min_3h").reset_index()

    step = binned["datetime"].diff().dropna().unique()
    assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
        f"dst 3h grid is not continuous: {step}"

    binned["dst_min48h"] = (
        binned["Dst_min_3h"].rolling(ROLL_BINS_48H, min_periods=1).min()
    )
    for mask_name, threshold in DST_MASKS.items():
        binned[mask_name] = binned["dst_min48h"] > threshold

    return binned


def report_survival(dst_3h: pd.DataFrame) -> pd.DataFrame:
    """full期間 (2024-04-01..2024-11-30) に絞った生存率レポート。"""
    period = dst_3h[
        (dst_3h["datetime"] >= PERIOD_START) & (dst_3h["datetime"] < PERIOD_END)
    ].copy()
    n_total = len(period)
    print(f"\n=== Dst quiet-mask survival (full period, {n_total} 3h bins, "
          f"{PERIOD_START.date()}..{PERIOD_END.date()}) ===")
    for mask_name in DST_MASKS:
        n_quiet = int(period[mask_name].sum())
        print(f"  {mask_name}: {n_quiet}/{n_total} bins "
              f"({100 * n_quiet / n_total:.1f}%) survive")

    OUT_MASK_3H.parent.mkdir(parents=True, exist_ok=True)
    period.to_csv(OUT_MASK_3H, index=False)
    print(f"wrote {OUT_MASK_3H}")
    return period


# =====================================================================
# proxy / EUV loading (same approach as 10_euv_lag_correlation.py)
# =====================================================================

def remove_outliers_mad(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """10_euv_lag_correlation.py と同じ modified z-score 外れ値除去。"""
    n_nan = int(df[col].isna().sum())
    if n_nan:
        print(f"  dropping {n_nan} NaN bin(s) in '{col}' (gap-adjacent) before MAD calc")
        df = df.dropna(subset=[col]).copy()

    x = df[col].to_numpy()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mod_z = 0.6745 * (x - med) / mad
    mask = np.abs(mod_z) > threshold
    removed = df.loc[mask, ["datetime", col]].copy()
    removed["mod_z"] = mod_z[mask]
    print(f"MAD outlier removal on '{col}' (n={len(df)}, threshold=|z_mod|>{threshold}):")
    if len(removed):
        for _, r in removed.iterrows():
            print(f"  removed {r['datetime']}  {col}={r[col]:.2f}  z_mod={r['mod_z']:+.2f}")
    else:
        print("  removed 0 points")
    return df.loc[~mask].copy()


def load_proxy() -> pd.DataFrame:
    df = pd.read_csv(IN_DEDT, parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = remove_outliers_mad(df, "dEdt_Jkg_per_day", OUTLIER_THRESHOLD)
    df["proxy_raw"] = -df["dEdt_Jkg_per_day"]
    return df[["datetime", "proxy_raw"]].reset_index(drop=True)


def smooth_proxy(proxy_raw: np.ndarray, window: int) -> np.ndarray:
    return savgol_filter(proxy_raw, window_length=window, polyorder=1)


class EuvLookup:
    """euv_3h.csv の連続3hグリッドを整数演算で直接参照する
    (10_euv_lag_correlation.py と同じ実装)。"""

    def __init__(self, euv_df: pd.DataFrame):
        self.start = euv_df["datetime"].iloc[0]
        self.n = len(euv_df)
        self.arrays = {
            wl: euv_df[f"irr_{wl}"].to_numpy(dtype=float) for wl in WAVELENGTHS
        }
        step = euv_df["datetime"].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
            f"euv_3h.csv is not a continuous 3h grid: {step}"

    def grid_pos(self, times: pd.Series) -> np.ndarray:
        offset_h = (times - self.start) / pd.Timedelta(hours=1)
        return np.rint(offset_h.to_numpy() / 3.0).astype(np.int64)

    def lag_matrix(self, wavelength: str, pos0: np.ndarray,
                   lags_h: list[int]) -> np.ndarray:
        arr = self.arrays[wavelength]
        lag_bins = np.asarray(lags_h, dtype=np.int64) // 3
        pos = pos0[:, None] - lag_bins[None, :]
        valid = (pos >= 0) & (pos < self.n)
        out = np.full(pos.shape, np.nan)
        out[valid] = arr[pos[valid]]
        return out


def load_euv() -> pd.DataFrame:
    df = pd.read_csv(IN_EUV, parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("datetime").reset_index(drop=True)


def fast_spearman(x: np.ndarray, y: np.ndarray, min_n: int = MIN_N):
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, n


def peak_from_curve(lags: np.ndarray, rs: np.ndarray):
    mask = lags >= PEAK_MIN_LAG_H
    sub_r = rs[mask]
    if np.all(np.isnan(sub_r)):
        return np.nan, np.nan
    idx = np.nanargmax(sub_r)
    return lags[mask][idx], sub_r[idx]


# =====================================================================
# Step 6-7: Dst-masked lag correlation scan
# =====================================================================

def run_lag_scan(dst_3h_full_period: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy = load_proxy()
    euv_df = load_euv()
    euv = EuvLookup(euv_df)

    dst_join = dst_3h_full_period[["datetime", "dst_min48h"] + list(DST_MASKS)]
    merged = pd.merge(proxy, dst_join, on="datetime", how="left")
    n_missing = merged["dst_min48h"].isna().sum()
    print(f"\nproxy rows: {len(merged)}, missing Dst-3h match: {n_missing}")
    if n_missing:
        print("  (dropping proxy rows without a matching dst_quiet_mask_3h bin, "
              "i.e. outside the full analysis period)")
        merged = merged.dropna(subset=["dst_min48h"]).reset_index(drop=True)
    for mask_name in DST_MASKS:
        merged[mask_name] = merged[mask_name].astype(bool)

    datetimes_all = merged["datetime"]
    proxy_raw_all = merged["proxy_raw"].to_numpy()
    proxy_smoothed = smooth_proxy(proxy_raw_all, SG_WINDOW)

    corr_rows = []
    for mask_name in DST_MASKS:
        gmask = merged[mask_name].to_numpy()
        sel_times = datetimes_all[gmask]
        sel_proxy = proxy_smoothed[gmask]
        n_sel = int(gmask.sum())
        print(f"  mask={mask_name}: {n_sel} proxy bins selected")
        pos0 = euv.grid_pos(sel_times)
        for wl in WAVELENGTHS:
            M = euv.lag_matrix(wl, pos0, LAGS_H)
            for li, L in enumerate(LAGS_H):
                r, n = fast_spearman(M[:, li], sel_proxy)
                corr_rows.append({
                    "mask": mask_name, "wavelength": wl,
                    "lag_hours": L, "spearman_r": r, "n": n,
                })

    corr = pd.DataFrame(corr_rows)
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT_CORR, index=False)
    print(f"\nwrote {OUT_CORR} ({len(corr)} rows)")

    peak_rows = []
    for (mask_name, wl), g in corr.groupby(["mask", "wavelength"], sort=False):
        lags = g["lag_hours"].to_numpy()
        rs = g["spearman_r"].to_numpy()
        peak_lag, peak_r = peak_from_curve(lags, rs)
        n_at_peak = g.loc[g["lag_hours"] == peak_lag, "n"]
        n_at_peak = int(n_at_peak.iloc[0]) if len(n_at_peak) and not np.isnan(peak_lag) else 0
        peak_rows.append({
            "mask": mask_name, "wavelength": wl,
            "peak_lag_hours": peak_lag, "peak_r": peak_r, "n": n_at_peak,
        })
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(OUT_PEAKS, index=False)
    print(f"wrote {OUT_PEAKS} ({len(peaks)} rows)")
    return corr, peaks


# =====================================================================
# Step 8: comparison table vs. Kp-based peaks
# =====================================================================

def build_comparison(dst_peaks: pd.DataFrame) -> pd.DataFrame:
    kp_peaks = pd.read_csv(IN_EUV_LAG_PEAKS)
    kp_sel = kp_peaks[
        (kp_peaks["window"] == 13) & (kp_peaks["period"] == "full") &
        (kp_peaks["geomag"].isin(["all", "quiet_kp4", "quiet_kp3"]))
    ].copy()
    kp_sel["wavelength"] = kp_sel["wavelength"].astype(str)

    kp_wide = kp_sel.pivot(index="wavelength", columns="geomag",
                            values="peak_lag_hours")
    kp_wide = kp_wide[["all", "quiet_kp4", "quiet_kp3"]]

    dst_wide = dst_peaks.pivot(index="wavelength", columns="mask",
                                values="peak_lag_hours")
    dst_wide = dst_wide[["quiet_dst50", "quiet_dst30"]]

    comparison = kp_wide.join(dst_wide, how="outer")
    comparison = comparison.reset_index()
    # wavelength列を数値順に並べ替え (256,284,304,1175,1216,1335,1405)
    order = {wl: i for i, wl in enumerate(WAVELENGTHS)}
    comparison["_order"] = comparison["wavelength"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")

    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_COMPARISON, index=False)
    print(f"\nwrote {OUT_COMPARISON}")

    print("\n=== Peak lag [h] comparison: Kp-based vs. Dst-based masks "
          "(window=13, period=full) ===")
    print(comparison.to_string(index=False))

    # 判定: EUV/FUV旧論文帯 (30-36h, 48-54h) の区別がKpとDstで同じ傾向を示すか
    print("\n=== Judgement ===")
    for _, row in comparison.iterrows():
        wl = row["wavelength"]
        vals = {k: row[k] for k in
                ["all", "quiet_kp4", "quiet_kp3", "quiet_dst50", "quiet_dst30"]
                if pd.notna(row.get(k, np.nan))}
        if len(vals) < 2:
            continue
        spread = max(vals.values()) - min(vals.values())
        print(f"  {wl:>5s} nm: " +
              ", ".join(f"{k}={v:.0f}h" for k, v in vals.items()) +
              f"   (spread across conditions = {spread:.0f}h)")

    return comparison


# =====================================================================
# Step 9: storm events (Kp>=6) for figure annotation
# =====================================================================

def storm_events_kp6() -> pd.DataFrame:
    df = pd.read_csv(IN_GEOMAG, parse_dates=["datetime"]).sort_values("datetime")
    df = df[(df["datetime"] >= PERIOD_START) & (df["datetime"] < PERIOD_END)]
    hot = df[df["Kp"] >= STORM_KP_THRESHOLD].sort_values("datetime")
    events = []
    if len(hot):
        times = hot["datetime"].to_numpy()
        kps = hot["Kp"].to_numpy()
        start_i = 0
        for i in range(1, len(times) + 1):
            if i == len(times) or times[i] - times[i - 1] > np.timedelta64(3, "h"):
                seg_t = times[start_i:i]
                seg_k = kps[start_i:i]
                pk = int(np.argmax(seg_k))
                events.append({
                    "start": pd.Timestamp(seg_t[0]),
                    "end": pd.Timestamp(seg_t[-1]) + BIN_TD,
                    "peak_kp": float(seg_k[pk]),
                    "peak_time": pd.Timestamp(seg_t[pk]),
                })
                start_i = i
    events_df = pd.DataFrame(events)
    OUT_STORM_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(OUT_STORM_EVENTS, index=False)
    print(f"\nwrote {OUT_STORM_EVENTS} ({len(events_df)} Kp>=6 events)")
    return events_df


# =====================================================================
# main
# =====================================================================

def main() -> None:
    dst_hourly_full = parse_omni2_dst()
    dst_3h = build_dst_3h(dst_hourly_full)
    dst_3h_full_period = report_survival(dst_3h)
    _corr, peaks = run_lag_scan(dst_3h_full_period)
    build_comparison(peaks)
    storm_events_kp6()
    print("\nDone.")


if __name__ == "__main__":
    main()
