"""プリホワイトニング(帯域通過)後のEUV遅延相関 — 自己相関バイアスを除いたラグ推定。

対応TODO: #1, #4, #7

背景:
  src/10 の生系列ラグ相関は、proxy・EUVの両方が持つ強い低周波自己相関
  (27日太陽自転・季節上昇トレンド) のため曲線が平坦なプラトーになり、
  argmaxが不安定 (docs/2026-07-02_euv_lag_correlation.md)。
  応答物理が住む1〜3日帯だけを残す帯域通過 (centered rolling mean の差:
  24h平滑 − 5日トレンド。centeredなので位相遅れなし) を両系列に適用して
  からラグ相関を取り、短期応答の遅延を自己相関バイアスなしで推定する。

入力:  data/processed/dEdt_3h.csv, euv_3h.csv
出力:  data/figure_data/prewhitened_lag_correlation.csv
       (wavelength, lag_hours, spearman_r, n)
実行:  .venv/Scripts/python src/11_prewhitened_lag.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_PROXY = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_EUV = ROOT / "data" / "processed" / "euv_3h.csv"
OUT = ROOT / "data" / "figure_data" / "prewhitened_lag_correlation.csv"

WAVELENGTHS = ["irr_256", "irr_284", "irr_304",
               "irr_1175", "irr_1216", "irr_1335", "irr_1405"]
HI_BINS = 8    # 24h 平滑 (3hビン×8)
LO_BINS = 40   # 5日 トレンド (3hビン×40)
MAX_LAG_BINS = 40  # 120h
MAD_Z = 8.0    # src/07 と同じ外れ値基準


def bandpass(s: pd.Series) -> pd.Series:
    sm = s.rolling(HI_BINS, center=True, min_periods=HI_BINS // 2).mean()
    tr = s.rolling(LO_BINS, center=True, min_periods=LO_BINS // 2).mean()
    return sm - tr


def main() -> None:
    d = pd.read_csv(IN_PROXY, parse_dates=["datetime"]).set_index("datetime")
    e = pd.read_csv(IN_EUV, parse_dates=["datetime"]).set_index("datetime")

    x = -d["dEdt_Jkg_per_day"]
    med = x.median()
    mad = (x - med).abs().median()
    x = x.mask(np.abs(0.6745 * (x - med) / mad) > MAD_Z)

    grid = pd.date_range(x.index.min(), x.index.max(), freq="3h", tz="UTC")
    xb = bandpass(x.reindex(grid))

    rows = []
    for wl in WAVELENGTHS:
        yb = bandpass(e[wl].reindex(grid))
        for k in range(0, MAX_LAG_BINS + 1):
            yy = yb.shift(k)
            ok = xb.notna() & yy.notna()
            r = spearmanr(xb[ok], yy[ok]).statistic
            rows.append({"wavelength": wl, "lag_hours": 3 * k,
                         "spearman_r": r, "n": int(ok.sum())})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    print("\n=== peak lag (lag >= 6h) per wavelength, band-passed ===")
    for wl, g in out.groupby("wavelength"):
        g6 = g[g.lag_hours >= 6]
        pk = g6.loc[g6.spearman_r.idxmax()]
        print(f"{wl:9s} peak {pk.lag_hours:5.0f}h  r={pk.spearman_r:+.3f}  (n={pk.n:.0f})")


if __name__ == "__main__":
    main()
