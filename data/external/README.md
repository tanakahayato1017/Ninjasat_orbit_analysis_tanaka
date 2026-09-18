# External data

Every external file used by the paper is included here; all of them are redistributable
under their respective terms, listed below.

| path | source | licence / terms | used by |
|---|---|---|---|
| `goes_euv/{YYMM}_csv_combined.csv` (2404 to 2411) | GOES-18 EUVS L2 1-min averages (`irr_256` .. `irr_1405` in W m^-2, `MgII_Index`, per-channel quality flags, `AU_Factor`), NOAA NCEI: https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance | US Government work, public domain | `09_prepare_euv.py`, `15_bin_size_sensitivity.py` |
| `tudelft/SA_DNS_POD_2024_{04..11}_v02.zip`, `SB_DNS_POD_..._v02.zip`, `GC_DNS_ACC_..._v02c.zip` | Swarm A, Swarm B (POD-derived) and GRACE-FO C (accelerometer-derived) thermospheric densities, TU Delft thermosphere data portal version 02: https://thermosphere.tudelft.nl/data/data/version_02/ (`Swarm_data/`, `GRACE-FO_data/`); method in van den IJssel et al. (2020), *Adv. Space Res.* | CC BY 4.0 | `06_tudelft_density.py` |
| `Kp_ap_since_1932.txt` | 3-hourly Kp and ap, GFZ Potsdam: https://kp.gfz.de/app/files/Kp_ap_since_1932.txt | CC BY 4.0 (Matzka et al. 2021) | `05_geomagnetic_indices.py` |
| `Kp_ap_Ap_SN_F107_since_1932.txt` | daily Ap, sunspot number, F10.7, GFZ Potsdam: https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt | CC BY 4.0 | `05_geomagnetic_indices.py` |
| `omni2/omni2_2024.dat`, `omni2/omni2.text` | OMNI2 hourly data for 2024 and its format description, NASA GSFC SPDF: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/ (Dst is word 41) | public; Dst originally from WDC for Geomagnetism, Kyoto | `18_dst_quiet_check.py` |

The GOES files are the NOAA L2 1-min product re-packaged as one CSV per month; the
`Time` column is UTC. Months 2407 to 2409 carry a few days of padding into the
neighbouring months, which `09_prepare_euv.py` removes by de-duplicating time stamps.

The TU Delft zip files are unmodified copies of the portal files;
`tudelft/download_log.txt` records the sizes retrieved on 2026-07-02. The 3-hourly and
daily products derived from them (`data/processed/tudelft_density_3h.csv`,
`tudelft_density_daily.csv`) are also included. Please credit TU Delft and cite
van den IJssel et al. (2020) when using them.
