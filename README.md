# Thermospheric Density Analysis: Solar EUV Variability and LEO CubeSat Orbit

This repository contains the analysis codes associated with the paper:

**"Thermospheric Expansion Driven by Solar EUV Variability: Impacts on the Orbit of a LEO CubeSat"**
Hayato Tanaka et al., *Earth, Planets and Space*, (2025, in preparation)

---

## Overview

This repository provides Python scripts for analyzing thermospheric density 
variations driven by solar EUV radiation and their impacts on the orbital 
decay of a low Earth orbit (LEO) CubeSat (NinjaSat).
The analysis combines GOES-18 EXIS EUV observations with GNSS-derived 
orbital data from NinjaSat.

---

## Contents

| Directory / File | Description |
|-----------------|-------------|
| `orbit/`        | Orbit propagation and decay analysis scripts (SGP4/TLE-based) |
| `euv/`          | GOES-18 EXIS EUV data processing and Savitzky-Golay filtering |
| `crosscorr/`    | Cross-correlation analysis between EUV flux and thermospheric density |
| `data/`         | Sample input data or data download scripts |
| `figures/`      | Scripts for reproducing figures in the paper |

---

## Requirements

- Python 3.9+
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

## Usage

```bash
# Orbit decay analysis
python orbit/propagate_tle.py --input data/ninjaSat_tle.txt

# EUV data processing
python euv/process_exis.py --input data/goes18_exis.nc

# Cross-correlation analysis
python crosscorr/xcorr_euv_density.py
```

---

## Citation

If you use this code, please cite:
