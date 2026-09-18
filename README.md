# NinjaSat orbital-energy decay analysis: code and data

Code and data supporting

> Tanaka H., Nakagawa T., Nozawa S., Tamagawa T., Enoto T., Kitaguchi T., Iwakiri W.,
> Mihara T., Takeda T., Ota N., Aoyama A., Iwata S., Takahashi T., Yamasaki K.
> **Thermospheric Response to Solar EUV Variability Probed with Orbital-Energy Decay
> from NinjaSat CubeSat GNSS Telemetry.**
> *Earth, Planets and Space* (under review, manuscript EPSP-D-26-00116).

The study derives a thermospheric-drag proxy from the on-board GNSS telemetry of the
NinjaSat CubeSat: the specific mechanical energy of the orbit including the J2 term,
$E_{J2}$, is computed from every GNSS fix, binned to 3 h, and its decay rate $-dE/dt$ is
validated against independent Swarm A/B and GRACE-FO densities and correlated with the
GOES-18 EUVS solar EUV/FUV irradiance at lags of 0 to 120 h.

This repository replaces the material published here for the original (2026-04)
submission, which used a TLE-minus-GNSS altitude-difference proxy. That material is
retained under [`data/legacy_v1/`](data/legacy_v1/README.md) for the record.

## Data policy in one paragraph

The NinjaSat GNSS telemetry is released with the permission of the NinjaSat team (RIKEN)
for the eleven columns listed in [`data/gnss/README.md`](data/gnss/README.md): time, fix
quality, geodetic position, satellite counts and geoid separation. **The ground-speed
column (`gpsGroundSpeed`) and the HDOP column are not included.** They are part of the
spacecraft housekeeping stream and their use requires explicit permission from RIKEN.
Because the kinetic-energy term of $E_{J2}$ is computed from the ground speed, the
per-sample energy products are likewise not released. What is released instead is every
product from the 3-hourly binning onwards (`data/processed/dEdt_3h.csv` and the figure
data), so that every validation, lag-correlation and sensitivity result in the paper can be
regenerated exactly. Researchers who need the velocity column should contact the
corresponding author (Hayato Tanaka, tanaka.h.eff6@m.isct.ac.jp), who will forward the
request to the NinjaSat team.

## Layout

```
data/
  gnss/         NinjaSat GNSS telemetry, public columns, 2024-03-31 .. 2024-11-30
  tle/          NinjaSat TLEs (Space-Track), used for the inclination and the legacy audits
  external/     GOES-18 EUVS irradiance, GFZ Kp/ap, NASA OMNI2 (Dst), TU Delft densities
                (all redistributable; sources and licences in data/external/README.md)
  processed/    3-hourly and daily products (dE/dt, EUV, Kp, density, Dst)
  figure_data/  the CSV behind every figure and every number quoted in the paper
  legacy_v1/    original-submission material (superseded method)
src/            analysis pipeline, numbered in execution order
plots/          figure scripts; they only read data/figure_data and write figures/
figures/        output directory (empty in the repository)
requirements.txt
```

## Reproducing the paper

Python 3.11 with the packages in `requirements.txt`. Run everything from the repository
root:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # or .venv/bin/pip on Linux/macOS
.venv\Scripts\python src\10_euv_lag_correlation.py  # etc.
```

### Pipeline

| # | script | input | output | paper |
|---|---|---|---|---|
| 01 | `01_prepare_gps.py` | `data/gnss/*.csv` | `processed/gps_qc.csv` | Sect. 2.1 |
| 02 | `02_specific_energy.py` | `gps_qc.csv` | `processed/specific_energy.csv` (**needs `gpsGroundSpeed`**, see above) | Sect. 2.2, Eq. 1, Fig. 1 |
| 03 | `03_dEdt_proxy.py` | `specific_energy.csv` | `processed/dEdt_3h.csv` (**supplied**) | Sect. 2.2 |
| 05 | `05_geomagnetic_indices.py` | GFZ Kp/ap files | `processed/geomag_3h.csv`, `geomag_daily.csv` | Sect. 2.4 |
| 06 | `06_tudelft_density.py` | `data/external/tudelft/*.zip` | `processed/tudelft_density_3h.csv`, `_daily.csv` (**supplied**) | Sect. 2.3 |
| 07 | `07_validate_dEdt.py` | 03 + 06 | `figure_data/dEdt_vs_density*.csv` | Sect. 3.2, Fig. 2 |
| 09 | `09_prepare_euv.py` | `data/external/goes_euv/*.csv` | `processed/euv_3h.csv` | Sect. 2.5 |
| 10 | `10_euv_lag_correlation.py` | 03 + 05 + 09 | `figure_data/euv_lag_*.csv` | Sect. 2.6, 3.3, Figs. 4, 5, 6 |
| 11 | `11_prewhitened_lag.py` | 03 + 09 | `figure_data/prewhitened_lag_correlation.csv` | Sect. 3.3 |
| 14 | `14_smoothing_symmetry.py` | 03 + 05 + 09 | `figure_data/smoothing_symmetry*.csv` | Sect. 2.7 (symmetric-smoothing test) |
| 15 | `15_bin_size_sensitivity.py` | `specific_energy.csv` (**restricted**) | `figure_data/bin_size_sensitivity.csv` (supplied) | Sect. 2.7 (bin-width sensitivity) |
| 16 | `16_diagnostics_figdata.py` | `specific_energy.csv` (**restricted**), legacy proxy | `figure_data/full_timeseries_euv_proxy.csv`, `proxy_comparison.csv` (supplied) | Fig. 3 |
| 17 | `17_qc_overview_figdata.py` | `gps_qc.csv` | `figure_data/gps_qc_overview_*.csv` | Sect. 2.1 (GNSS QC summary) |
| 17 | `17_altitude_semantics.py` | `gps_qc.csv` + TLE | `figure_data/altitude_semantics*.csv` | Sect. 3.1 (altitude-interpretation test) |
| 17 | `17_E_full_timeseries_figdata.py`, `17_despike_casestudy_figdata.py` | `specific_energy.csv` (**restricted**) | supplied | Sect. 2.2, 2.7 (despiking diagnostics) |
| 18 | `18_dst_quiet_check.py` | 03 + 05 + 09 + OMNI2 | `figure_data/dst_*.csv` | Sect. 2.4, 3.3 (Dst screening) |
| 19 | `19_srp_gravity_bound.py` | `specific_energy.csv` (**restricted**) | console (`figure_data/error_budget_bounds.csv` supplied) | Sect. 2.7 (error budget) |
| 24 | `24_gnss_error_decomposition.py` | `specific_energy.csv` (**restricted**) | `figure_data/gnss_error_decomposition.csv` (supplied) | Sect. 2.6, 2.7 |
| 12, 13, 20, 21 | legacy-method audits | `data/legacy_v1/`, TLE, GNSS | `figure_data/old_*`, `tle_epoch_*`, `legacy_S*` | not in the paper (reviewer correspondence) |
| 04, 08 | legacy-method audits | `specific_energy.csv` (**restricted**), legacy proxy | `figure_data/proxy_comparison*`, `sign_audit_*` (supplied) | not in the paper (reviewer correspondence) |

Scripts marked **restricted** stop with a `FileNotFoundError` for
`data/processed/specific_energy.csv`: that per-sample energy file cannot be built
without `gpsGroundSpeed`. Their outputs are included in `data/figure_data/` so that the
plots and all downstream statistics still reproduce. Scripts 01, 12 and 21 run on the
public file; in place of the speed-window QC step they drop, by time stamp, the 11 fixes
that failed that window on the full telemetry, so the selected sample is identical to the
paper's.

### Figures

| paper figure | script | figure_data input |
|---|---|---|
| Fig. 1 | `plots/plot_energy_conservation.py` | `energy_vs_altitude_conservation.csv` |
| Fig. 2 | `plots/plot_dEdt_vs_density.py` | `dEdt_vs_density.csv`, `dEdt_vs_density_stats.csv` |
| Fig. 3 | `plots/plot_full_timeseries_euv.py` | `full_timeseries_euv_proxy.csv` |
| Fig. 4 | `plots/plot_euv_lag_scan.py` | `euv_lag_correlation.csv`, `euv_lag_peaks.csv` |
| Fig. 5 | `plots/plot_bootstrap_ci.py` | `euv_lag_bootstrap.csv` |
| Fig. 6 | `plots/plot_figure6_combined.py` | `smoothing_symmetry*.csv`, `euv_lag_peaks.csv` |
| diagnostics | remaining `plots/plot_*.py` | as named in each script docstring |

### Sensitivity analyses and diagnostics referenced in the paper

| item in the paper (Availability of data and materials) | script | figure data |
|---|---|---|
| symmetric-smoothing test | `14_smoothing_symmetry.py` | `smoothing_symmetry.csv`, `smoothing_symmetry_peaks.csv` |
| bin-width sensitivity | `15_bin_size_sensitivity.py` | `bin_size_sensitivity.csv` |
| prewhitened lag correlation | `11_prewhitened_lag.py` | `prewhitened_lag_correlation.csv` |
| Dst-based geomagnetic screening | `18_dst_quiet_check.py` | `dst_quiet_check.csv`, `dst_quiet_peaks.csv`, `dst_kp_peak_comparison.csv` |
| sample-level despiking diagnostics | `02_specific_energy.py`, `17_despike_casestudy_figdata.py` | `despike_casestudy.csv` |
| GNSS telemetry quality-control summary | `01_prepare_gps.py`, `17_qc_overview_figdata.py` | `gps_qc_overview_*.csv` |
| altitude-interpretation reconstruction test (Sect. 3.1) | `17_altitude_semantics.py` | `altitude_semantics.csv`, `altitude_semantics_stats.csv` |
| GNSS error decomposition, 12-h ripple origin (Sect. 2.6, 2.7) | `24_gnss_error_decomposition.py` | `gnss_error_decomposition.csv` |

Script docstrings are written in Japanese; each states inputs, outputs and the method in
the order the code executes.

## Data sources and credits

- NinjaSat GNSS telemetry and TLEs: NinjaSat team, RIKEN; TLEs via Space-Track.org
  (18 SPCS).
- GOES-18 EUVS L2 1-min irradiance: NOAA NCEI / SWPC.
- Kp, ap, Ap: GFZ Helmholtz Centre for Geosciences, Potsdam (CC BY 4.0).
- Dst (OMNI2): NASA GSFC SPDF OMNIWeb, originally WDC for Geomagnetism, Kyoto.
- Swarm A/B and GRACE-FO thermospheric densities: TU Delft thermosphere data portal,
  version 02.

See [`data/external/README.md`](data/external/README.md) for URLs and retrieval steps.

## Citing

Please cite the paper above. A machine-readable citation is in `CITATION.cff`.
