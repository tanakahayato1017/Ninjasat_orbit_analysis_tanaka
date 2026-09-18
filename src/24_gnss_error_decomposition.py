# -*- coding: utf-8 -*-
"""E散らばりのGNSS測定誤差分解と、無平滑ラグ曲線リップルの起源診断。

教授コメント(2026-07-16)への対応:
 (a) 「E_J2の日内ばらつき(519 J/kg)はGNSSの測定誤差から説明できるか」
     -> 隣接サンプル(~600 s)差分で白色成分を分離し、等価速度/高度誤差に換算。
 (b) 「無平滑のラグ相関曲線(Fig 4a)の周期的・波長間相関する残差の原因は」
     -> プロキシ(-dE/dt 3hビン)のパワースペクトルの半日周期(12.00 h)線を定量化し、
        リップル残差の波長間相関がプロキシ共有によるコモンモードであることを確認。

出力: data/figure_data/gnss_error_decomposition.csv, 標準出力に要約統計。
"""
import numpy as np
import pandas as pd
from scipy import signal

PROC = 'data/processed'
OUT = 'data/figure_data/gnss_error_decomposition.csv'

rows = []


def add(k, v, note=''):
    rows.append({'key': k, 'value': v, 'note': note})
    print(f'{k:42s} {v:12.4g}  {note}')


# ---------------------------------------------------------------- (a) E scatter
se = pd.read_csv(f'{PROC}/specific_energy.csv', parse_dates=['datetime'])
se = se.sort_values('datetime').reset_index(drop=True)
t = se['datetime']
E = se['E_J2_Jkg'].values
day = t.dt.floor('D')

wd = se.groupby(day)['E_J2_Jkg'].std().dropna()
sig_day = float(np.median(wd))
add('within_day_std_EJ2_Jkg', sig_day, 'median over days (despiked)')

# adjacent-sample (~600 s) first differences isolate the white (sample-level)
# noise component: secular drag over 600 s is only ~4-6 J/kg, negligible.
dt_s = t.diff().dt.total_seconds().values
dE = np.diff(E, prepend=np.nan)
m = (dt_s > 550) & (dt_s < 650)
add('n_adjacent_pairs', int(m.sum()))
sig_white = float(np.nanstd(dE[m]) / np.sqrt(2))
add('white_noise_per_sample_Jkg', sig_white, 'std(adjacent dE)/sqrt(2)')
add('white_variance_share', sig_white**2 / sig_day**2,
    'fraction of within-day variance that is white')
sig_struct = float(np.sqrt(max(sig_day**2 - sig_white**2, 0.0)))
add('structured_component_Jkg', sig_struct, 'sqrt(within-day^2 - white^2)')

v_med = float(se['v_inertial_ms'].median())
g_eff = 3.986004418e14 / float(se['r_m'].median())**2
add('equiv_speed_error_cms', sig_white / v_med * 100, 'dE/dv = v_i')
add('equiv_height_error_m', sig_white / g_eff, 'dE/dh = mu/r^2 (if all in position)')

# latitude-locked structure (detrend per day by median, bin by latitude)
se['E_detr'] = se['E_J2_Jkg'] - se.groupby(day)['E_J2_Jkg'].transform('median')
bins = np.arange(-90, 90.1, 15)
latpat = se.groupby(pd.cut(se['lat_deg'], bins), observed=True)['E_detr'].mean()
add('lat_pattern_peak_to_peak_Jkg', float(latpat.max() - latpat.min()),
    '15-deg-bin means of day-detrended E')

# ------------------------------------------------- (b) semidiurnal line / ripple
de = pd.read_csv(f'{PROC}/dEdt_3h.csv', parse_dates=['datetime']).set_index('datetime')
eu = pd.read_csv(f'{PROC}/euv_3h.csv', parse_dates=['datetime']).set_index('datetime')


def line_strength(x, p0, nperseg=1024):
    """spectral peak near period p0 [h] relative to the local continuum."""
    x = pd.Series(x).interpolate(limit=8).dropna().values
    x = x - x.mean()
    f, P = signal.welch(x, fs=1 / 3.0, nperseg=min(nperseg, len(x)))
    per, Pn = 1 / f[1:], P[1:]
    band = (per > p0 * 0.92) & (per < p0 * 1.1)
    cont = (per > p0 * 0.6) & (per < p0 * 1.67) & ~band
    return float(Pn[band].max() / np.median(Pn[cont])), float(per[band][Pn[band].argmax()])

r12, p12 = line_strength(de['dEdt_Jkg_per_day'], 12.0)
add('proxy_12h_line_over_continuum', r12, f'peak period {p12:.2f} h')
r24, _ = line_strength(de['dEdt_Jkg_per_day'], 24.0)
add('proxy_24h_line_over_continuum', r24)
for c in ['irr_256', 'irr_284', 'irr_304', 'irr_1175', 'irr_1216', 'irr_1335', 'irr_1405']:
    r, _ = line_strength(eu[c], 12.0)
    add(f'euv_{c}_12h_line_over_continuum', r)

# cross-wavelength correlation of the ripple residuals in the unsmoothed
# lag-correlation curves (window = -1 means no smoothing)
lc = pd.read_csv('data/figure_data/euv_lag_correlation.csv')
sub = lc[(lc.window == -1) & (lc.geomag == 'all') & (lc.period == 'full')]
piv = sub.pivot(index='lag_hours', columns='wavelength', values='spearman_r').sort_index()
res = piv - piv.rolling(11, center=True, min_periods=5).mean()
C = res.corr().values
add('ripple_mean_crosswavelength_corr', float(C[np.triu_indices(len(piv.columns), 1)].mean()),
    'expected: shared proxy -> common-mode ripple')

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'\nwrote {OUT}')
