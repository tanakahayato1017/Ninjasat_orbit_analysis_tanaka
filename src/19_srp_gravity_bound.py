"""太陽輻射圧(SRP)・重力モデル近似(高次項/第三体)が (-dE/dt) proxy に及ぼす
誤差の上界を定量評価する。

対応TODO: #6 (不確かさの定量化), R2 Major#1
  (「SGP4-GNSS残差にはSRP・重力モデル近似等も混入しうる。無視できることを示せ」
  への回答材料。ただし本解析は新proxy (-dE/dt, E_J2ベース) 自身に対する評価であり
  SGP4を経由しない)

入力:
  data/processed/dEdt_3h.csv        (03_dEdt_proxy.py の出力。減衰シグナル・
                                       統計誤差se_dEdtの実測値を読む)
  data/processed/specific_energy.csv (02_specific_energy.py の出力、despike後。
                                       E_J2の日内std・日次サンプル数N_dayを実測する)

出力:
  data/figure_data/error_budget_bounds.csv
      列: item, worst_case_Jkg_day, pct_of_signal_low_pct, pct_of_signal_typical_pct, note

手法:
  1. SRP worst-case (物理的に最悪 = 軌道内で常に速度方向に加速すると仮定, 非物理的
     だが厳密な上界):
       a_SRP = (Phi/c)(1+q)(A/m),  Phi=1361 W/m^2, c=3e8 m/s, q=0.3 (保守的反射係数)
       |dE/dt|_SRP,worst <= a_SRP * v * 86400   [J/kg/day],  v=7.68 km/s (円軌道速さ)
     A/m は NinjaSat (6U CubeSat, 質量8.0 kg [Enoto et al. 2020 SPIE; Enoto et al.
     2024, arXiv:2412.03016 / PASJ 2025, mass 8/8.14 kg, 100x200x300 mm stowed])
     の断面積レンジでスキャンする。デプロイ後の太陽電池パドル面積の一次資料値は
     見つからなかったため、タスク仕様のフォールバック
     (断面積 0.02-0.06 m^2, stowed felt面〜最大格納面) に加え、デプロイ後の
     保守的な上振れケース (0.10, 0.15 m^2) も追加でスキャンする。
  2. 実際には1軌道の前半・後半でSRPの符号 (速度に対する成分) がほぼ反転し
     大部分が相殺する (SRPは保存力に近い非散逸的摂動)。正味の非零成分は
     食(eclipse)による対称性の破れ由来のみで、経験的にworst-caseの
     ~10-30%程度 (係数0.1-0.3、文献的な目安。本解析では上界の頑健性を
     示すため、この係数をかけても結論が変わらないことを確認する)。
  3. 重力高次項 (J3以上)・月太陽第三体は保存力 (ポテンシャル力) であり、
     単独では長期的なエネルギー散逸を生まない (軌道周期・月/太陽周期の
     有界な周期振動のみを作る)。したがって「無視できるか」は主に
     (a) 軌道内変動として観測量にどれだけ混入するか (= despike後のE_J2の
     日内std、これは高次重力項・速度近似誤差・GNSS計測誤差の合計を含む
     経験的上界) と (b) 日次差分・日平均への漏れ込みがどれだけ小さいか
     で評価する:
       日次平均への寄与 se_grav,day ~ std_E_J2,day / sqrt(N_day)
       日次差分 (連続2日) への寄与 ~ sqrt(2) * se_grav,day  [J/kg/day]
     を実測値 (std_E_J2,day の全日median, N_dayの全日median) から計算する。

実行:
  .venv/Scripts/python src/19_srp_gravity_bound.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DEDT = ROOT / "data" / "processed" / "dEdt_3h.csv"
IN_ENERGY = ROOT / "data" / "processed" / "specific_energy.csv"
OUT_CSV = ROOT / "data" / "figure_data" / "error_budget_bounds.csv"

# --- 物理定数 ---
PHI = 1361.0            # 太陽定数 [W/m^2]
C_LIGHT = 3.0e8          # 光速 [m/s]
Q_REFLECT = 0.3          # 反射係数 (保守的、q=0で完全吸収、q=1で完全鏡面反射)
V_ORBIT = 7.68e3         # NinjaSat 円軌道速さの目安 [m/s] (~500-530km, ~7.6km/s)
SECONDS_PER_DAY = 86400.0

# --- NinjaSat諸元 (WebSearchで確認した一次資料値) ---
# Enoto et al. 2020, SPIE Proc. 11444 ("NinjaSat: an agile CubeSat approach for
#   monitoring of bright x-ray compact objects") — 計画時の6U CubeSatとして
#   質量8kg、6U(100x200x300mm)を記載。
# Enoto et al. 2024/2025, arXiv:2412.03016 -> PASJ 77, 466 (2025)
#   ("NinjaSat: Astronomical X-ray CubeSat Observatory") — 打上げ後の実機として
#   質量8kg(概要)/8.14kg(詳細表)、6U(100x200x300mm, 格納時)、太陽電池パドルは
#   格納状態で打上げ後に軌道上で展開(打上げ当日に展開完了)、軌道高度約519-530km、
#   軌道傾斜角97度(太陽同期極軌道)と記載。デプロイ後のパドル面積の数値は
#   両文献とも明記していない。
NINJASAT_MASS_KG = 8.0  # 8.0kgと8.14kgの差は1.7%程度でworst-case評価に影響しない

# A/m スキャン用の断面積レンジ [m^2]
#   0.02, 0.06: タスク仕様のフォールバックレンジ (6Uの最小面 0.1x0.2m と
#               最大格納面 0.2x0.3m にほぼ対応する stowed 状態の目安)
#   0.10, 0.15: デプロイ後の太陽電池パドルを見込んだ保守的な上振れケース
#               (一次資料に数値なし、格納時最大面の1.7-2.5倍を仮置き)
AREA_SCAN_M2 = [0.02, 0.06, 0.10, 0.15]

# 食(eclipse)非対称性による正味SRP寄与の目安係数 (worst-caseに対する比)
NET_FACTOR_LOW = 0.1
NET_FACTOR_HIGH = 0.3


def compute_srp_bounds() -> pd.DataFrame:
    rows = []
    for area in AREA_SCAN_M2:
        am = area / NINJASAT_MASS_KG
        a_srp = (PHI / C_LIGHT) * (1.0 + Q_REFLECT) * am
        worst_Jkg_day = a_srp * V_ORBIT * SECONDS_PER_DAY
        net_low = worst_Jkg_day * NET_FACTOR_LOW
        net_high = worst_Jkg_day * NET_FACTOR_HIGH
        rows.append({
            "area_m2": area,
            "A_over_m_m2_per_kg": am,
            "a_srp_m_s2": a_srp,
            "srp_worst_case_Jkg_day": worst_Jkg_day,
            "srp_net_eclipse_low_Jkg_day": net_low,
            "srp_net_eclipse_high_Jkg_day": net_high,
        })
    return pd.DataFrame(rows)


def compute_gravity_bound() -> dict:
    energy = pd.read_csv(IN_ENERGY, parse_dates=["datetime"])
    day = energy["datetime"].dt.floor("D")
    grp = energy.groupby(day)
    daily_std = grp["E_J2_Jkg"].std()      # J/kg, 日内std (despike後)
    daily_n = grp["E_J2_Jkg"].size()        # 日内サンプル数

    std_med = float(daily_std.median())
    n_med = float(daily_n.median())

    se_daily_mean = std_med / np.sqrt(n_med)          # J/kg, 日平均Eへの寄与
    se_daily_diff = np.sqrt(2.0) * se_daily_mean       # J/kg/day, 連続2日差分への寄与

    return {
        "std_E_J2_daily_median_Jkg": std_med,
        "N_day_median": n_med,
        "se_daily_mean_Jkg": se_daily_mean,
        "se_daily_diff_Jkg_day": se_daily_diff,
    }


def load_signal_and_statistical_error() -> dict:
    dedt = pd.read_csv(IN_DEDT, parse_dates=["datetime"])
    valid = dedt.dropna(subset=["dEdt_Jkg_per_day"])
    daily = valid.set_index("datetime").resample("1D")["dEdt_Jkg_per_day"].mean().dropna()
    daily_abs = daily.abs()

    se_dedt_3h_med = float(valid["se_dEdt_Jkg_per_day"].median())
    # 日平均への統計誤差伝播 (docs/2026-07-02_pipeline_remediation.md §3.6と同じロジック):
    # se_daily ~= se_E_median / (N_bins_per_day * dt_day)
    se_e_med = float(valid["se_E_Jkg"].median())
    n_bins_per_day = 8.0
    dt_day = 3.0 / 24.0
    se_dedt_daily = se_e_med / (n_bins_per_day * dt_day)

    return {
        "signal_p10_Jkg_day": float(np.percentile(daily_abs, 10)),
        "signal_median_Jkg_day": float(np.percentile(daily_abs, 50)),
        "signal_p90_Jkg_day": float(np.percentile(daily_abs, 90)),
        "se_dEdt_3h_median_Jkg_day": se_dedt_3h_med,
        "se_dEdt_daily_Jkg_day": se_dedt_daily,
    }


def main() -> None:
    srp_df = compute_srp_bounds()
    grav = compute_gravity_bound()
    sig = load_signal_and_statistical_error()

    signal_low = sig["signal_p10_Jkg_day"]      # 保守的な比較対象 (小さい方のシグナル)
    signal_typ = sig["signal_median_Jkg_day"]

    print("=== signal & statistical-error reference (from data) ===")
    print(f"daily |dE/dt| : p10={sig['signal_p10_Jkg_day']:.1f}  "
          f"median={sig['signal_median_Jkg_day']:.1f}  "
          f"p90={sig['signal_p90_Jkg_day']:.1f}  [J/kg/day]")
    print(f"se_dEdt (3h bin, median)  = {sig['se_dEdt_3h_median_Jkg_day']:.1f} J/kg/day")
    print(f"se_dEdt (daily, propagated) = {sig['se_dEdt_daily_Jkg_day']:.1f} J/kg/day")

    print("\n=== SRP worst-case scan (A/m) ===")
    print(f"NinjaSat mass = {NINJASAT_MASS_KG} kg  "
          f"(Enoto et al. 2020 SPIE; Enoto et al. 2024/2025 arXiv:2412.03016 / PASJ)")
    for _, r in srp_df.iterrows():
        pct_worst_low = 100.0 * r["srp_worst_case_Jkg_day"] / signal_low
        pct_worst_typ = 100.0 * r["srp_worst_case_Jkg_day"] / signal_typ
        pct_net_low = 100.0 * r["srp_net_eclipse_high_Jkg_day"] / signal_low
        print(f"  A={r['area_m2']:.2f} m^2  A/m={r['A_over_m_m2_per_kg']:.5f} m^2/kg  "
              f"a_SRP={r['a_srp_m_s2']:.3e} m/s^2  "
              f"worst-case={r['srp_worst_case_Jkg_day']:.2f} J/kg/day "
              f"({pct_worst_low:.2f}% of p10 signal, {pct_worst_typ:.2f}% of median signal)  "
              f"net(eclipse, {NET_FACTOR_LOW}-{NET_FACTOR_HIGH}x)="
              f"{r['srp_net_eclipse_low_Jkg_day']:.2f}-{r['srp_net_eclipse_high_Jkg_day']:.2f} "
              f"J/kg/day ({pct_net_low:.3f}% of p10 signal at high end)")

    print("\n=== gravity higher-order / third-body empirical bound ===")
    print(f"median daily std of E_J2 (despiked) = {grav['std_E_J2_daily_median_Jkg']:.1f} J/kg")
    print(f"median N_day (samples/day)          = {grav['N_day_median']:.0f}")
    print(f"-> daily-mean contribution se ~= std/sqrt(N) = "
          f"{grav['se_daily_mean_Jkg']:.1f} J/kg")
    print(f"-> daily-difference contribution ~= sqrt(2)*se = "
          f"{grav['se_daily_diff_Jkg_day']:.1f} J/kg/day")
    pct_grav_low = 100.0 * grav["se_daily_diff_Jkg_day"] / signal_low
    pct_grav_typ = 100.0 * grav["se_daily_diff_Jkg_day"] / signal_typ
    print(f"   = {pct_grav_low:.2f}% of p10 signal, {pct_grav_typ:.2f}% of median signal")

    # --- 出力: error_budget_bounds.csv ---
    rows = []

    # SRP: 断面積スキャンの代表点 (0.02, 0.06 = タスク既定レンジ、
    # 0.10, 0.15 = デプロイ後想定の上振れ) をそれぞれ1行ずつ記録
    for _, r in srp_df.iterrows():
        rows.append({
            "item": f"SRP_worst_case_A{r['area_m2']:.2f}m2",
            "worst_case_Jkg_day": r["srp_worst_case_Jkg_day"],
            "pct_of_signal_low_pct": 100.0 * r["srp_worst_case_Jkg_day"] / signal_low,
            "pct_of_signal_typical_pct": 100.0 * r["srp_worst_case_Jkg_day"] / signal_typ,
            "note": (f"physically-worst-case bound (always accelerating along "
                     f"velocity), A/m={r['A_over_m_m2_per_kg']:.4f} m^2/kg, "
                     f"m={NINJASAT_MASS_KG} kg, q={Q_REFLECT}"),
        })
    # 代表 (中間の面積0.06 m^2, stowed最大面) の食非対称性補正後ネット値
    mid = srp_df[srp_df["area_m2"] == 0.06].iloc[0]
    rows.append({
        "item": "SRP_net_eclipse_asymmetry_A0.06m2",
        "worst_case_Jkg_day": mid["srp_net_eclipse_high_Jkg_day"],
        "pct_of_signal_low_pct": 100.0 * mid["srp_net_eclipse_high_Jkg_day"] / signal_low,
        "pct_of_signal_typical_pct": 100.0 * mid["srp_net_eclipse_high_Jkg_day"] / signal_typ,
        "note": (f"non-dissipative SRP mostly cancels over an orbit; net residual "
                 f"from eclipse asymmetry taken as {NET_FACTOR_LOW}-{NET_FACTOR_HIGH}x "
                 "worst-case (upper end reported here)"),
    })

    rows.append({
        "item": "gravity_higher_order_thirdbody_daily_mean",
        "worst_case_Jkg_day": grav["se_daily_mean_Jkg"],
        "pct_of_signal_low_pct": 100.0 * grav["se_daily_mean_Jkg"] / signal_low,
        "pct_of_signal_typical_pct": 100.0 * grav["se_daily_mean_Jkg"] / signal_typ,
        "note": ("conservative forces (J3+, lunisolar 3rd-body) cannot create "
                 "secular energy loss; empirical bound = despiked E_J2 daily std "
                 f"({grav['std_E_J2_daily_median_Jkg']:.1f} J/kg, median) / "
                 f"sqrt(N_day={grav['N_day_median']:.0f}); bounds ALL unmodelled "
                 "periodic terms + velocity-approx error jointly"),
    })
    rows.append({
        "item": "gravity_higher_order_thirdbody_daily_diff",
        "worst_case_Jkg_day": grav["se_daily_diff_Jkg_day"],
        "pct_of_signal_low_pct": 100.0 * grav["se_daily_diff_Jkg_day"] / signal_low,
        "pct_of_signal_typical_pct": 100.0 * grav["se_daily_diff_Jkg_day"] / signal_typ,
        "note": "sqrt(2) x daily_mean bound, propagated to a 2-day difference "
                "(the scale at which the decay signal is read off)",
    })

    rows.append({
        "item": "statistical_error_se_dEdt_3h_median",
        "worst_case_Jkg_day": sig["se_dEdt_3h_median_Jkg_day"],
        "pct_of_signal_low_pct": 100.0 * sig["se_dEdt_3h_median_Jkg_day"] / signal_low,
        "pct_of_signal_typical_pct": 100.0 * sig["se_dEdt_3h_median_Jkg_day"] / signal_typ,
        "note": "median se_dEdt of single 3h-bin central difference "
                "(src/03_dEdt_proxy.py, bin-internal std propagation)",
    })
    rows.append({
        "item": "statistical_error_se_dEdt_daily",
        "worst_case_Jkg_day": sig["se_dEdt_daily_Jkg_day"],
        "pct_of_signal_low_pct": 100.0 * sig["se_dEdt_daily_Jkg_day"] / signal_low,
        "pct_of_signal_typical_pct": 100.0 * sig["se_dEdt_daily_Jkg_day"] / signal_typ,
        "note": "se_E_median / (N_bins_per_day * dt_day), see "
                "docs/2026-07-02_pipeline_remediation.md sec 3.6",
    })

    rows.append({
        "item": "decay_signal_reference_daily_|dEdt|",
        "worst_case_Jkg_day": signal_typ,
        "pct_of_signal_low_pct": 100.0 * signal_low / signal_low,
        "pct_of_signal_typical_pct": 100.0,
        "note": f"daily |dE/dt| distribution: p10={signal_low:.0f}, "
                f"median={signal_typ:.0f}, p90={sig['signal_p90_Jkg_day']:.0f} "
                "J/kg/day (this row is the reference scale, not an error term)",
    })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out)} rows)")

    print("\n=== error budget summary table ===")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(out[["item", "worst_case_Jkg_day", "pct_of_signal_low_pct",
                    "pct_of_signal_typical_pct"]].to_string(index=False,
                                                             float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
