"""平滑化の対称化感度: proxy側のみ平滑化する現行手法(非対称)と、EUV側にも同一の
Savitzky-Golay窓を適用した対称手法・両方無平滑の3条件でラグ相関ピークがどう動くか
を比較する。

対応TODO: #20 (自己相関と非対称平滑化の影響の明確化, R1)

背景:
  10_euv_lag_correlation.py はproxy(-dE/dt)側にのみSG平滑化(polyorder=1)をかけ、
  EUV側は3hビン平均のみ (平滑化なし) でラグ相関を取っている。R1は「両系列の
  平滑化条件が非対称であることがクロス相関 (ピークラグ・形状) に与える影響を
  明確化せよ」と指摘している。本スクリプトはEUV側にも同一のSG13窓
  (13点 ≈ 39h, polyorder=1) を適用した「対称」条件と、「両方無平滑」条件を
  追加し、現行の「非対称」条件と3方向で比較する。

入力:
  data/processed/dEdt_3h.csv   (03_dEdt_proxy.py の出力, 平滑化なし)
  data/processed/euv_3h.csv    (09_prepare_euv.py の出力, 連続3hグリッド)
  data/processed/geomag_3h.csv (05_geomagnetic_indices.py の出力)

手法:
  1. proxy = -dEdt_Jkg_per_day。10_euv_lag_correlation.py と同じ modified
     z-score (Iglewicz & Hoaglin) |z_mod| > 8.0 の外れ値除去を dEdt_Jkg_per_day
     に適用する (2024-10-25の2ビンが対象)。欠測隣接NaN (03の改修方針2) は
     中央値/MAD計算の前にあらかじめ除外する。
  2. SG平滑化 (window=13点=39h, polyorder=1) は「有効値のみを時系列順に並べた
     コンパクトな系列」に対してかける (savgol_filter は等間隔仮定・NaN非対応の
     ため)。これは 03_dEdt_proxy.py / 10_euv_lag_correlation.py がproxy側で
     既に採用している近似と同じ (欠測ギャップを跨ぐと厳密には時間軸が歪むが、
     本解析の全条件で同一の近似を使うため条件間の比較は公平)。
     EUV側もwavelengthごとに同じ方式で扱う: euv_3h.csv の連続3hグリッドのうち
     NaNでない値だけを抜き出してSG13をかけ、結果を元のグリッド位置に戻す
     (NaNだったビンはNaNのまま = 平滑化によって欠測を埋めない)。
  3. 3条件:
       asymmetric (現行10相当): proxy=SG13, EUV=raw
       symmetric              : proxy=SG13, EUV=SG13 (同一窓)
       none                   : proxy=raw,  EUV=raw
  4. ラグ相関: L = 0, 3, ..., 120 [h] について
       spearman_r = Spearman corr( EUV(t-L), proxy(t) )
     10_euv_lag_correlation.py と同じ整数グリッド参照 (線形補間なし)。
     マスク (地磁気条件) は常にproxy側の時刻 t に適用する。
  5. 地磁気条件:
       all       : フィルタなし
       quiet_kp4 : proxyビン t を含む直近48h (t-45h〜t, 16ビン) のKp最大値 < 4
  6. 期間: full = 2024-04-01〜2024-11-30 のみ (dEdt_3h.csvの2024-03-31パディング
     分は除外)。
  7. ピーク: 各 (condition, geomag, wavelength) につき lag_hours>=6 の範囲で
     spearman_r が最大となるラグ (10_euv_lag_correlation.py と同じ定義)。

出力:
  data/figure_data/smoothing_symmetry.csv
      columns: condition, geomag, wavelength, lag_hours, spearman_r, n
  data/figure_data/smoothing_symmetry_peaks.csv
      columns: condition, geomag, wavelength, peak_lag_hours, peak_r, n,
               delta_peak_vs_asymmetric_h (symmetric/none の場合のみ非NaN;
               asymmetric基準からのピークラグの変化量 [h])

判定 (コンソールに出力):
  対称化 (asymmetric -> symmetric) でピークラグが ±6h 以上動くかどうかを
  wavelength x geomag ごとに判定し、動いた組合せの一覧を表示する。

実行:
  .venv/Scripts/python src/14_smoothing_symmetry.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
IN_DEDT = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_EUV = ROOT / "data" / "processed" / "euv_3h.csv"
IN_GEOMAG = ROOT / "data" / "processed" / "geomag_3h.csv"
OUT_CORR = ROOT / "data" / "figure_data" / "smoothing_symmetry.csv"
OUT_PEAKS = ROOT / "data" / "figure_data" / "smoothing_symmetry_peaks.csv"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
GEOMAG_CONDS = ["all", "quiet_kp4"]
LAGS_H = list(range(0, 121, 3))             # 0,3,...,120 (41 lags)
PEAK_MIN_LAG_H = 6                          # lag<6h の自明な同時相関は除外

SG_WINDOW = 13                              # 13点 SG (polyorder=1) ~= 39h
OUTLIER_THRESHOLD = 8.0                     # 07/10 と同じMAD基準 (proxy外れ値除去)
MIN_N = 10
KP_ROLL_BINS = 16                           # 3h x 16 = 48h trailing window

CONDITIONS = ["asymmetric", "symmetric", "none"]
PEAK_DELTA_THRESHOLD_H = 6.0


# ---------------------------------------------------------------- proxy ----

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


def sg_compact(values: np.ndarray, window: int) -> np.ndarray:
    """有効値 (非NaN) のみをコンパクトに並べてSG平滑化し、元の位置に戻す。

    NaNだった位置はNaNのまま (平滑化で欠測を埋めない)。savgol_filter は
    NaN非対応・等間隔仮定のため、proxy/EUVとも同じ近似を採用する
    (欠測ギャップを跨ぐと時間軸はわずかに歪むが、全条件で同一の近似)。
    """
    out = np.full(values.shape, np.nan)
    mask = ~np.isnan(values)
    n_valid = int(mask.sum())
    if n_valid < window:
        # 有効点がwindow未満なら平滑化できないので生値をそのまま使う
        out[mask] = values[mask]
        return out
    out[mask] = savgol_filter(values[mask], window_length=window, polyorder=1)
    return out


# ------------------------------------------------------------------ EUV ----

class EuvLookup:
    """euv_3h.csv の連続3hグリッドを整数演算で直接参照する (10と同じ設計)。"""

    def __init__(self, euv_df: pd.DataFrame):
        self.start = euv_df["datetime"].iloc[0]
        self.n = len(euv_df)
        raw = {wl: euv_df[f"irr_{wl}"].to_numpy(dtype=float) for wl in WAVELENGTHS}
        self.arrays = {"raw": raw, "smoothed": {wl: sg_compact(raw[wl], SG_WINDOW) for wl in WAVELENGTHS}}
        step = euv_df["datetime"].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
            f"euv_3h.csv is not a continuous 3h grid: {step}"

    def grid_pos(self, times: pd.Series) -> np.ndarray:
        offset_h = (times - self.start) / pd.Timedelta(hours=1)
        return np.rint(offset_h.to_numpy() / 3.0).astype(np.int64)

    def lag_matrix(self, wavelength: str, euv_mode: str, pos0: np.ndarray,
                   lags_h: list[int]) -> np.ndarray:
        arr = self.arrays[euv_mode][wavelength]
        lag_bins = np.asarray(lags_h, dtype=np.int64) // 3
        pos = pos0[:, None] - lag_bins[None, :]           # (n_sel, n_lags)
        valid = (pos >= 0) & (pos < self.n)
        out = np.full(pos.shape, np.nan)
        out[valid] = arr[pos[valid]]
        return out


def load_euv() -> pd.DataFrame:
    df = pd.read_csv(IN_EUV, parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("datetime").reset_index(drop=True)


# --------------------------------------------------------------- geomag ----

def load_geomag_kp_max48h() -> pd.DataFrame:
    df = pd.read_csv(IN_GEOMAG, parse_dates=["datetime"]).sort_values("datetime")
    df = df.reset_index(drop=True)
    step = df["datetime"].diff().dropna().unique()
    assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
        f"geomag_3h.csv is not a continuous 3h grid: {step}"
    df["kp_max48h"] = df["Kp"].rolling(KP_ROLL_BINS, min_periods=1).max()
    return df[["datetime", "kp_max48h"]]


# ------------------------------------------------------------ correlation --

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


def period_bounds_full():
    return pd.Timestamp("2024-04-01"), pd.Timestamp("2024-12-01")


def geomag_mask(cond: str, kp_max48h: np.ndarray) -> np.ndarray:
    if cond == "all":
        return np.ones(len(kp_max48h), dtype=bool)
    if cond == "quiet_kp4":
        return kp_max48h < 4.0
    raise ValueError(cond)


def peak_from_curve(lags: np.ndarray, rs: np.ndarray):
    mask = lags >= PEAK_MIN_LAG_H
    sub_r = rs[mask]
    if np.all(np.isnan(sub_r)):
        return np.nan, np.nan
    idx = np.nanargmax(sub_r)
    return lags[mask][idx], sub_r[idx]


# CONDITION -> (proxy_mode, euv_mode)
CONDITION_MODES = {
    "asymmetric": ("smoothed", "raw"),
    "symmetric": ("smoothed", "smoothed"),
    "none": ("raw", "raw"),
}


# ---------------------------------------------------------------- main -----

def main() -> None:
    proxy = load_proxy()
    euv_df = load_euv()
    euv = EuvLookup(euv_df)
    kp = load_geomag_kp_max48h()

    merged = pd.merge(proxy, kp, on="datetime", how="left")
    n_missing_kp = merged["kp_max48h"].isna().sum()
    print(f"\nproxy rows: {len(merged)}, missing Kp match: {n_missing_kp}")
    if n_missing_kp:
        print("  (dropping proxy rows without a matching geomag_3h bin)")
        merged = merged.dropna(subset=["kp_max48h"]).reset_index(drop=True)

    kp_max48h_all = merged["kp_max48h"].to_numpy()
    datetimes_all = merged["datetime"]
    proxy_raw_all = merged["proxy_raw"].to_numpy()
    proxy_smoothed_all = sg_compact(proxy_raw_all, SG_WINDOW)

    proxy_arrays = {"raw": proxy_raw_all, "smoothed": proxy_smoothed_all}

    pstart, pend = period_bounds_full()
    pmask = ((datetimes_all >= pstart) & (datetimes_all < pend)).to_numpy()

    # ============================ ラグ相関 ============================
    corr_rows = []
    for condition in CONDITIONS:
        proxy_mode, euv_mode = CONDITION_MODES[condition]
        proxy_series = proxy_arrays[proxy_mode]
        for geomag in GEOMAG_CONDS:
            gmask = geomag_mask(geomag, kp_max48h_all)
            sel = pmask & gmask
            if sel.sum() < MIN_N:
                continue
            sel_times = datetimes_all[sel]
            sel_proxy = proxy_series[sel]
            pos0 = euv.grid_pos(sel_times)
            for wl in WAVELENGTHS:
                M = euv.lag_matrix(wl, euv_mode, pos0, LAGS_H)
                for li, L in enumerate(LAGS_H):
                    r, n = fast_spearman(M[:, li], sel_proxy)
                    corr_rows.append({
                        "condition": condition, "geomag": geomag,
                        "wavelength": wl, "lag_hours": L,
                        "spearman_r": r, "n": n,
                    })
    corr = pd.DataFrame(corr_rows)
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT_CORR, index=False)
    print(f"\nwrote {OUT_CORR} ({len(corr)} rows, "
          f"{len(CONDITIONS)} conditions x {len(GEOMAG_CONDS)} geomag x "
          f"{len(WAVELENGTHS)} wavelengths x {len(LAGS_H)} lags)")

    # ============================ ピーク表 ============================
    peak_rows = []
    peak_lookup = {}
    for (condition, geomag, wl), g in corr.groupby(
            ["condition", "geomag", "wavelength"], sort=False):
        lags = g["lag_hours"].to_numpy()
        rs = g["spearman_r"].to_numpy()
        peak_lag, peak_r = peak_from_curve(lags, rs)
        n_at_peak = g.loc[g["lag_hours"] == peak_lag, "n"]
        n_at_peak = int(n_at_peak.iloc[0]) if len(n_at_peak) and not np.isnan(peak_lag) else 0
        peak_lookup[(condition, geomag, wl)] = peak_lag
        peak_rows.append({
            "condition": condition, "geomag": geomag, "wavelength": wl,
            "peak_lag_hours": peak_lag, "peak_r": peak_r, "n": n_at_peak,
        })

    for row in peak_rows:
        base = peak_lookup.get(("asymmetric", row["geomag"], row["wavelength"]))
        if row["condition"] == "asymmetric" or base is None or np.isnan(base) \
                or np.isnan(row["peak_lag_hours"]):
            row["delta_peak_vs_asymmetric_h"] = np.nan
        else:
            row["delta_peak_vs_asymmetric_h"] = row["peak_lag_hours"] - base

    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(OUT_PEAKS, index=False)
    print(f"wrote {OUT_PEAKS} ({len(peaks)} rows)")

    # ============================ 判定 ============================
    print("\n=== judgement: |peak lag shift| >= "
          f"{PEAK_DELTA_THRESHOLD_H:.0f}h from asymmetric (current method) ===")
    print(f"{'geomag':10s} {'wl':6s} {'asym[h]':>8s} {'sym[h]':>8s} {'d_sym':>7s}  "
          f"{'none[h]':>8s} {'d_none':>7s}")
    n_shifted_sym = 0
    n_shifted_none = 0
    n_total = 0
    for geomag in GEOMAG_CONDS:
        for wl in WAVELENGTHS:
            asym = peak_lookup.get(("asymmetric", geomag, wl), np.nan)
            sym = peak_lookup.get(("symmetric", geomag, wl), np.nan)
            none = peak_lookup.get(("none", geomag, wl), np.nan)
            d_sym = sym - asym if not (np.isnan(sym) or np.isnan(asym)) else np.nan
            d_none = none - asym if not (np.isnan(none) or np.isnan(asym)) else np.nan
            n_total += 1
            flag_sym = "  <-- >=6h" if (not np.isnan(d_sym) and abs(d_sym) >= PEAK_DELTA_THRESHOLD_H) else ""
            if flag_sym:
                n_shifted_sym += 1
            if not np.isnan(d_none) and abs(d_none) >= PEAK_DELTA_THRESHOLD_H:
                n_shifted_none += 1
            print(f"{geomag:10s} {wl:6s} {asym:8.0f} {sym:8.0f} {d_sym:+7.0f}  "
                  f"{none:8.0f} {d_none:+7.0f}{flag_sym}")
    print(f"\nsymmetric vs asymmetric: {n_shifted_sym}/{n_total} combos shifted by "
          f">= {PEAK_DELTA_THRESHOLD_H:.0f}h")
    print(f"none vs asymmetric     : {n_shifted_none}/{n_total} combos shifted by "
          f">= {PEAK_DELTA_THRESHOLD_H:.0f}h")


if __name__ == "__main__":
    main()
