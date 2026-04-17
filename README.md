# Thermospheric Density Analysis: Solar EUV Variability and LEO CubeSat Orbit

This repository contains the analysis codes associated with the paper:

**"Thermospheric Expansion Driven by Solar EUV Variability: Impacts on the Orbit of a LEO CubeSat"**
Hayato Tanaka et al., *Earth, Planets and Space*, (2026, in preparation)

---

## Overview

This repository provides Python scripts for analyzing thermospheric density 
variations driven by solar EUV radiation and their impacts on the orbital 
decay of a low Earth orbit (LEO) CubeSat (NinjaSat).
The analysis combines GOES-18 EXIS EUV observations with GNSS-derived 
orbital data from NinjaSat.

---

## Contents

| Directory / File          | Description |
|--------------------------|-------------|
| `comparison_GNSS_TLE/`   | Comparison scripts between GNSS-derived and TLE-propagated orbits |
| `correlation/`           | Cross-correlation analysis between solar EUV flux and thermospheric density |
| `Difference_gps_SGP4/`   | Analysis of orbital altitude differences between GPS observations and SGP4 propagation |
| `example_altitude_fitting/` | Example scripts for altitude fitting procedures |
| `GNSS_data/`             | GNSS-derived orbital data from NinjaSat |
| `Savitzky-golay/`        | Savitzky-Golay smoothing filter applied to the rate of change of orbital altitude time series |

---

## Requirements

- Python 3.11.8+
- numpy
- scipy
- astropy
- matplotlib
- sgp4

Install dependencies:

```bash
pip install numpy scipy astropy matplotlib sgp4
```

---

## Data Sources

- **GOES-18 EXIS EUV**: [NOAA NCEI](https://www.ncei.noaa.gov/)
- **NinjaSat TLE / GNSS data**: ISAS/JAXA
- **Geomagnetic indices (Kp, Dst)**: [GFZ Potsdam](https://www.gfz-potsdam.de/)


---

## Citation

If you use this code, please cite:
