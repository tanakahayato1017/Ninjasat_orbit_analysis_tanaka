"""EUV/FUV各波長 vs (-dE/dt) proxy のラグ相関スキャン
(SG平滑化窓・地磁気条件・期間の全組合せ)。

対応TODO: #4 (Savitzky-Golayフィルタ39h窓のアーティファクト検証),
          #8 (地磁気活動の定量的評価),
          #1 (軌道減衰指標の再検討 — dE/dt proxyでの遅延相関再計算),
          #6 (不確かさの定量化 — ブートストラップCI)

入力:
  data/processed/dEdt_3h.csv   (03_dEdt_proxy.py の出力, 平滑化なし)
  data/processed/euv_3h.csv    (09_prepare_euv.py の出力, 連続3hグリッド)
  data/processed/geomag_3h.csv (05_geomagnetic_indices.py の出力)

手法:
  1. proxy = -dEdt_Jkg_per_day。外れ値除去は 07_validate_dEdt.py と同じ
     modified z-score (Iglewicz & Hoaglin) |z_mod| > 8.0 を dEdt_Jkg_per_day
     (3h系列) に適用する。現行の主データセットでは 03_dEdt_proxy.py の
     上流despike (改修方針1) により2024-10-25の既知異常は除去済みで、
     このスクリーンが除去するビンは0個 (安全策として保持)。
     03_dEdt_proxy.py の欠測跨ぎ差分対策 (改修方針2) により dEdt_Jkg_per_day
     には少数のNaNビン (欠測隣接) が含まれるため、中央値/MAD計算の前に
     あらかじめ除外する (07と同じ扱い)。
  2. SG平滑化 (polyorder=1) は proxy 系列 (-dEdt_Jkg_per_day, 外れ値除去後の
     compact な系列) に直接かける。03_dEdt_proxy.py はE系列を平滑化してから
     微分するが、本スクリプトはTODO #4の感度分析として「窓幅を変えたときに
     ラグ相関のピークがどう動くか」を見るのが目的なので、dE/dt系列自体への
     平滑化として実装する (03とは平滑化を掛ける段が異なる点に注意。
     いずれも「39h窓相当のSG平滑化がピーク位置を作っていないか」を検証する
     という目的は同じ)。savgol_filter は等間隔仮定のため、dEdt_3h.csv内の
     欠測ギャップ (最大9日程度、23箇所) をまたぐと時間軸がわずかに歪む
     (03/07と同じ既知の制約)。
  3. ラグ相関: L = 0, 3, ..., 120 [h] について
       spearman_r = Spearman corr( EUV(t-L), proxy(t) )
     EUV(t-L) は euv_3h.csv の連続3hグリッド上を時刻ベースで直接参照する
     (整数グリッド演算、線形補間はしない)。EUV側を過去にずらす = 「EUV変動が
     先行し、proxyがL時間後に応答する」という向き。
     マスク (地磁気条件・期間) は常に proxy側の時刻 t に対して適用する
     (EUV側には適用しない)。
  4. 地磁気条件:
       all       : フィルタなし
       quiet_kp4 : proxyビン t を含む直近48h (t-45h〜t, 16ビン) のKp最大値 < 4
       quiet_kp3 : 同 < 3
     「直近48hの最大Kp」は熱圏の嵐応答が数日残ることを踏まえ、単純な瞬時Kpでは
     なく trailing 48h window の最大値で判定する。
  5. 期間: full=2024-04-01〜2024-11-30 (dEdt_3h.csvの2024-03-31パディング分は
     除外), 月別 2404..2411 (各暦月)。
  6. ピーク: 各 (wavelength, window, geomag, period) につき lag_hours>=6 の範囲で
     spearman_rが最大となるラグ (lag 0-3hの自明な同時相関を除外するため)。
     全ラグの生カーブは出力1にすべて残す。
  7. ブロックブートストラップ: full期間、window∈{なし,13}、geomag∈{all,
     quiet_kp4} の組合せ (7波長 x 2 x 2 = 28通り) のみ。5日の非重複ブロックに
     区切り (2024-04-01を起点)、ブロックを重複ありで再抽出して系列を組み替え、
     500回のブートストラップ標本それぞれでピークラグ (定義は6と同じ) を記録、
     その分布のpercentile 2.5/16/50/84/97.5を出力する。seed=42固定。

出力:
  data/figure_data/euv_lag_correlation.csv
      columns: wavelength, window, geomag, period, lag_hours, spearman_r, n
      (window は -1 = 平滑化なし, それ以外はSG点数)
  data/figure_data/euv_lag_peaks.csv
      columns: wavelength, window, geomag, period, peak_lag_hours, peak_r, n
  data/figure_data/euv_lag_bootstrap.csv
      columns: wavelength, window, geomag, p2_5, p16, p50, p84, p97_5,
               n_boot_valid (500点中ピークが定義できた回数)

実行:
  .venv/Scripts/python src/10_euv_lag_correlation.py
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
OUT_CORR = ROOT / "data" / "figure_data" / "euv_lag_correlation.csv"
OUT_PEAKS = ROOT / "data" / "figure_data" / "euv_lag_peaks.csv"
OUT_BOOT = ROOT / "data" / "figure_data" / "euv_lag_bootstrap.csv"

WAVELENGTHS = ["256", "284", "304", "1175", "1216", "1335", "1405"]
WINDOWS = [None, 5, 9, 13, 21, 39]          # SG polyorder=1, None=平滑化なし
GEOMAG_CONDS = ["all", "quiet_kp4", "quiet_kp3"]
MONTHS = ["2404", "2405", "2406", "2407", "2408", "2409", "2410", "2411"]
PERIODS = ["full"] + MONTHS
LAGS_H = list(range(0, 121, 3))             # 0,3,...,120 (41 lags)
PEAK_MIN_LAG_H = 6                          # lag<6h の自明な同時相関は除外

OUTLIER_THRESHOLD = 8.0                     # 07_validate_dEdt.py と同じMAD基準
MIN_N = 10                                  # これ未満は相関を計算しない (NaN)
KP_ROLL_BINS = 16                           # 3h x 16 = 48h trailing window

N_BOOT = 500
BOOT_BLOCK_DAYS = 5
BOOT_SEED = 42
BOOT_WINDOWS = [None, 13]
BOOT_GEOMAG = ["all", "quiet_kp4"]


def window_label(w):
    return -1 if w is None else w


# ---------------------------------------------------------------- proxy ----

def remove_outliers_mad(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """07_validate_dEdt.py と同じ modified z-score 外れ値除去。

    dEdt_Jkg_per_day の欠測隣接NaN (03_dEdt_proxy.py 改修方針2) は
    中央値/MAD計算の前にあらかじめ除外する。
    """
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


def smooth_proxy(proxy_raw: np.ndarray, window: int | None) -> np.ndarray:
    if window is None:
        return proxy_raw
    return savgol_filter(proxy_raw, window_length=window, polyorder=1)


# ------------------------------------------------------------------ EUV ----

class EuvLookup:
    """euv_3h.csv の連続3hグリッドを整数演算で直接参照する。"""

    def __init__(self, euv_df: pd.DataFrame):
        self.start = euv_df["datetime"].iloc[0]
        self.n = len(euv_df)
        self.arrays = {
            wl: euv_df[f"irr_{wl}"].to_numpy(dtype=float) for wl in WAVELENGTHS
        }
        # グリッド間隔チェック (すべて3hで等間隔であることを保証)
        step = euv_df["datetime"].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta("3h"), \
            f"euv_3h.csv is not a continuous 3h grid: {step}"

    def grid_pos(self, times: pd.Series) -> np.ndarray:
        offset_h = (times - self.start) / pd.Timedelta(hours=1)
        return np.rint(offset_h.to_numpy() / 3.0).astype(np.int64)

    def lag_matrix(self, wavelength: str, pos0: np.ndarray,
                   lags_h: list[int]) -> np.ndarray:
        """M[i, li] = EUV(sel_times[i] - lags_h[li])。範囲外はNaN。

        (ベクトル化した整数グリッド参照。時刻→グリッド位置の変換は grid_pos で
        1回だけ行い、全ラグ分を fancy indexing でまとめて取り出す。)
        """
        arr = self.arrays[wavelength]
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


def period_bounds(period: str):
    if period == "full":
        return pd.Timestamp("2024-04-01"), pd.Timestamp("2024-12-01")
    year, month = 2024, int(period[2:])
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.DateOffset(months=1))
    return start, end


def geomag_mask(cond: str, kp_max48h: np.ndarray) -> np.ndarray:
    if cond == "all":
        return np.ones(len(kp_max48h), dtype=bool)
    if cond == "quiet_kp4":
        return kp_max48h < 4.0
    if cond == "quiet_kp3":
        return kp_max48h < 3.0
    raise ValueError(cond)


def block_ids(times: pd.Series, ref_start: pd.Timestamp, block_days: int) -> np.ndarray:
    return ((times - ref_start) / pd.Timedelta(days=block_days)).astype(np.int64).to_numpy()


def peak_from_curve(lags: np.ndarray, rs: np.ndarray):
    mask = lags >= PEAK_MIN_LAG_H
    sub_r = rs[mask]
    if np.all(np.isnan(sub_r)):
        return np.nan, np.nan
    idx = np.nanargmax(sub_r)
    return lags[mask][idx], sub_r[idx]


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

    # --- window ごとに平滑化した proxy 系列 (compact, chronological) ---
    smoothed_by_window = {w: smooth_proxy(proxy_raw_all, w) for w in WINDOWS}

    # ================= 出力1・2: 全組合せのラグ相関 & ピーク =================
    corr_rows = []
    n_combos = len(WINDOWS) * len(PERIODS) * len(GEOMAG_CONDS)
    combo_i = 0
    for w in WINDOWS:
        proxy_smoothed = smoothed_by_window[w]
        for period in PERIODS:
            pstart, pend = period_bounds(period)
            pmask = (datetimes_all >= pstart) & (datetimes_all < pend)
            for geomag in GEOMAG_CONDS:
                gmask = geomag_mask(geomag, kp_max48h_all)
                sel = (pmask.to_numpy()) & gmask
                combo_i += 1
                if sel.sum() < MIN_N:
                    continue
                sel_times = datetimes_all[sel]
                sel_proxy = proxy_smoothed[sel]
                pos0 = euv.grid_pos(sel_times)
                for wl in WAVELENGTHS:
                    M = euv.lag_matrix(wl, pos0, LAGS_H)
                    for li, L in enumerate(LAGS_H):
                        r, n = fast_spearman(M[:, li], sel_proxy)
                        corr_rows.append({
                            "wavelength": wl, "window": window_label(w),
                            "geomag": geomag, "period": period,
                            "lag_hours": L, "spearman_r": r, "n": n,
                        })
    corr = pd.DataFrame(corr_rows)
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT_CORR, index=False)
    print(f"\nwrote {OUT_CORR} ({len(corr)} rows, "
          f"{n_combos} window x period x geomag combos x "
          f"{len(WAVELENGTHS)} wavelengths x {len(LAGS_H)} lags)")

    peak_rows = []
    for (wl, w, geomag, period), g in corr.groupby(
            ["wavelength", "window", "geomag", "period"], sort=False):
        lags = g["lag_hours"].to_numpy()
        rs = g["spearman_r"].to_numpy()
        peak_lag, peak_r = peak_from_curve(lags, rs)
        n_at_peak = g.loc[g["lag_hours"] == peak_lag, "n"]
        n_at_peak = int(n_at_peak.iloc[0]) if len(n_at_peak) and not np.isnan(peak_lag) else 0
        peak_rows.append({
            "wavelength": wl, "window": w, "geomag": geomag, "period": period,
            "peak_lag_hours": peak_lag, "peak_r": peak_r, "n": n_at_peak,
        })
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(OUT_PEAKS, index=False)
    print(f"wrote {OUT_PEAKS} ({len(peaks)} rows)")

    # ============================ 出力3: ブートストラップ ============================
    print(f"\n=== block bootstrap (N={N_BOOT}, block={BOOT_BLOCK_DAYS}d, seed={BOOT_SEED}) ===")
    boot_rows = []
    ref_start = period_bounds("full")[0]
    pstart, pend = period_bounds("full")
    pmask = (datetimes_all >= pstart) & (datetimes_all < pend)

    for w in BOOT_WINDOWS:
        proxy_smoothed = smoothed_by_window[w]
        for geomag in BOOT_GEOMAG:
            gmask = geomag_mask(geomag, kp_max48h_all)
            sel = (pmask.to_numpy()) & gmask
            sel_times = datetimes_all[sel].reset_index(drop=True)
            sel_proxy = proxy_smoothed[sel]
            pos0 = euv.grid_pos(sel_times)
            bids = block_ids(sel_times, ref_start, BOOT_BLOCK_DAYS)
            unique_blocks = np.unique(bids)
            block_to_idx = {b: np.where(bids == b)[0] for b in unique_blocks}

            for wl in WAVELENGTHS:
                # ラグ行列は選択集合に対して1回だけ作り、ブートストラップの
                # 各反復では行の再サンプリング (fancy indexing) だけを行う。
                M = euv.lag_matrix(wl, pos0, LAGS_H)
                rng = np.random.default_rng(BOOT_SEED)
                peak_lags = []
                for _ in range(N_BOOT):
                    chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                    idx = np.concatenate([block_to_idx[b] for b in chosen])
                    Mb = M[idx]
                    boot_proxy = sel_proxy[idx]
                    rs = np.full(len(LAGS_H), np.nan)
                    for li in range(len(LAGS_H)):
                        r, n = fast_spearman(Mb[:, li], boot_proxy)
                        rs[li] = r
                    peak_lag, _ = peak_from_curve(np.array(LAGS_H), rs)
                    if not np.isnan(peak_lag):
                        peak_lags.append(peak_lag)
                peak_lags = np.array(peak_lags)
                n_valid = len(peak_lags)
                if n_valid == 0:
                    pcts = [np.nan] * 5
                else:
                    pcts = np.percentile(peak_lags, [2.5, 16, 50, 84, 97.5])
                boot_rows.append({
                    "wavelength": wl, "window": window_label(w), "geomag": geomag,
                    "p2_5": pcts[0], "p16": pcts[1], "p50": pcts[2],
                    "p84": pcts[3], "p97_5": pcts[4], "n_boot_valid": n_valid,
                })
                print(f"  window={window_label(w):>3} geomag={geomag:10s} "
                      f"wl={wl:5s}  peak_lag median={pcts[2]:.1f}h "
                      f"[{pcts[0]:.1f}, {pcts[4]:.1f}] (n_valid={n_valid}/{N_BOOT})")

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT_BOOT, index=False)
    print(f"\nwrote {OUT_BOOT} ({len(boot)} rows)")


if __name__ == "__main__":
    main()
