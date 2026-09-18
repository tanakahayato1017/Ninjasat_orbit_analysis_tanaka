"""補足図S1-S3 (旧手法, original submission相当) の描画。

対応TODO: #12, 査読対応の最終図整理
manuscripts.tex 461-465行目のSupplementary Figure S1/S2/S3キャプションに対応。

入力:
  data/figure_data/legacy_S1_raw_2404.csv   (21_legacy_figures_S1S3.py の出力,
      2024-04-29/30 生カデンスGPS/TLE高度)
  data/figure_data/legacy_S1_bins_2404.csv  (同, 2軌道ビンの正弦フィットD)
  data/figure_data/legacy_S2S3_monthly_2407.csv  (同, 2024-07アーカイブ月次値
      + 平滑化微分)

出力:
  figures/FigureS1.png / .pdf   1日分の1周回正弦フィット例 (2024-04-29/30)
  figures/FigureS2.png / .pdf   2024-07月間のTLE/GNSS高度・差分 (2軸)
  figures/FigureS3.png / .pdf   2024-07月間の高度差分と平滑化微分 (2軸)

スライドタイトル文字列は一切含めない (develop branchのFigure1-3.pdfはPowerPoint
書き出しでスライドタイトルが残存していたため, 本図は生データから独立に再描画する)。

すべて図データCSVから描く (再計算しない)。
実行:  .venv/Scripts/python plots/plot_legacy_S1S3.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_S1_RAW = ROOT / "data" / "figure_data" / "legacy_S1_raw_2404.csv"
IN_S1_BINS = ROOT / "data" / "figure_data" / "legacy_S1_bins_2404.csv"
IN_S2S3 = ROOT / "data" / "figure_data" / "legacy_S2S3_monthly_2407.csv"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 13
FS_TITLE = 15
FS_TEXT = 12
FS_PANEL = 16

GPS_COLOR = "#3355cc"
TLE_COLOR = "#cc3333"
DIFF_COLOR = "#2ca02c"
DERIV_COLOR = "#ff8c00"


# ============================================================================
# S1: 1日分の1周回正弦フィット例 (2024-04-29/30)
# ============================================================================
def plot_s1() -> None:
    raw = pd.read_csv(IN_S1_RAW, parse_dates=["datetime"])
    bins = pd.read_csv(IN_S1_BINS, parse_dates=["bin_center"])

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(raw["datetime"], raw["h_gps_km"], "--", lw=1.0, color=GPS_COLOR,
            alpha=0.75, label="GPS altitude (raw, ~600 s cadence)")
    ax.plot(raw["datetime"], raw["h_tle_km"], "--", lw=1.0, color=TLE_COLOR,
            alpha=0.75, label="TLE (SGP4) altitude (raw, ~600 s cadence)")

    ax.plot(bins["bin_center"], bins["D_gps_km"], "o--", ms=7, lw=1.3,
            color=GPS_COLOR, mec="black", mew=0.6, zorder=5,
            label="GPS average altitude (2-orbit bin fit)")
    ax.plot(bins["bin_center"], bins["D_tle_km"], "o--", ms=7, lw=1.3,
            color=TLE_COLOR, mec="black", mew=0.6, zorder=5,
            label="TLE average altitude (2-orbit bin fit)")

    ax.set_xlabel("UTC, 2024-04-29 to 2024-04-30", fontsize=FS_LABEL)
    ax.set_ylabel("Altitude [km]" "\n" r"($|\mathbf{r}| - R_{\mathrm{eq}}$, legacy definition)",
                  fontsize=FS_LABEL)
    ax.set_title("Legacy proxy: sinusoidal fitting procedure (one-day example)",
                 fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", ncol=1)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    fig.text(0.01, 0.01,
              "note: GPS/TLE candidate definitions and 2-orbit-bin IQR-filtered 1/rev sinusoid fit "
              "reproduced from the audited legacy-code definition\n"
              "(src/12_old_altitude_definition_audit.py; docs/2026-07-02_old_altitude_definition.md)",
              ha="left", va="bottom", fontsize=FS_TEXT - 2, color="gray")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT = OUT_DIR / "FigureS1"
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=160)
    print(f"wrote {OUT}.pdf / .png")
    plt.close(fig)


# ============================================================================
# S2: 2024-07月間のTLE伝播高度・GNSS高度・差分 (2軸)
# ============================================================================
def plot_s2() -> None:
    df = pd.read_csv(IN_S2S3, parse_dates=["timestamp"])

    fig, ax1 = plt.subplots(figsize=(12, 7.2))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df["timestamp"], df["gps_average_altitude"], "o-", ms=3, lw=0.9,
                    color=GPS_COLOR, alpha=0.85, label="GPS average altitude")
    l2, = ax1.plot(df["timestamp"], df["tle_average_altitude"], "o-", ms=3, lw=0.9,
                    color=TLE_COLOR, alpha=0.85, label="TLE average altitude")
    l3, = ax2.plot(df["timestamp"], df["altitude_diff"], "-", lw=1.3,
                    color=DIFF_COLOR, alpha=0.9, label="TLE $-$ GPS altitude difference")

    ax1.set_xlabel("UTC, July 2024", fontsize=FS_LABEL)
    ax1.set_ylabel("Altitude [km]", fontsize=FS_LABEL, color="0.15")
    ax2.set_ylabel(r"Altitude difference $\Delta h$ [km]", fontsize=FS_LABEL, color=DIFF_COLOR)
    ax2.tick_params(axis="y", labelcolor=DIFF_COLOR, labelsize=FS_TICK)
    ax1.tick_params(axis="y", labelsize=FS_TICK)
    ax1.tick_params(axis="x", labelsize=FS_TICK)
    ax1.set_title("Legacy proxy: monthly comparison of TLE-propagated and "
                  "GNSS-derived altitude (July 2024)", fontsize=FS_TITLE, pad=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.grid(alpha=0.3)

    lines = [l1, l2, l3]
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=FS_LEGEND,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT = OUT_DIR / "FigureS2"
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=160)
    print(f"wrote {OUT}.pdf / .png")
    plt.close(fig)


# ============================================================================
# S3: 2024-07月間の高度差分時系列とその平滑化微分 (2軸)
# ============================================================================
def plot_s3() -> None:
    df = pd.read_csv(IN_S2S3, parse_dates=["timestamp"])
    deriv_km_per_day = df["ddeltahdt_km_per_s"] * 86400.0

    fig, ax1 = plt.subplots(figsize=(12, 7.2))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df["timestamp"], df["altitude_diff"], "-", lw=1.4,
                    color=DIFF_COLOR, alpha=0.9, label=r"Altitude difference $\Delta h$ [km]")
    l2, = ax2.plot(df["timestamp"], deriv_km_per_day, "-", lw=1.1,
                    color=DERIV_COLOR, alpha=0.85,
                    label=r"$\widehat{\Delta \dot h}(t)$ (SG window=13, ~39h) [km day$^{-1}$]")

    ax1.set_xlabel("UTC, July 2024", fontsize=FS_LABEL)
    ax1.set_ylabel(r"Altitude difference $\Delta h$ [km]", fontsize=FS_LABEL, color=DIFF_COLOR)
    ax2.set_ylabel(r"Smoothed rate of change $\widehat{\Delta \dot h}(t)$"
                   "\n[km day$^{-1}$]", fontsize=FS_LABEL, color=DERIV_COLOR)
    ax1.tick_params(axis="y", labelcolor=DIFF_COLOR, labelsize=FS_TICK)
    ax2.tick_params(axis="y", labelcolor=DERIV_COLOR, labelsize=FS_TICK)
    ax1.tick_params(axis="x", labelsize=FS_TICK)
    ax1.axhline(0, color="gray", lw=0.5, zorder=0)
    ax2.axhline(0, color=DERIV_COLOR, lw=0.5, ls=":", alpha=0.5, zorder=0)
    ax1.set_title("Legacy proxy: altitude-difference time series and its "
                  "smoothed rate of change (July 2024)", fontsize=FS_TITLE, pad=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.grid(alpha=0.3)

    lines = [l1, l2]
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=FS_LEGEND,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.24))

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT = OUT_DIR / "FigureS3"
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=160)
    print(f"wrote {OUT}.pdf / .png")
    plt.close(fig)


def main() -> None:
    plot_s1()
    plot_s2()
    plot_s3()


if __name__ == "__main__":
    main()
