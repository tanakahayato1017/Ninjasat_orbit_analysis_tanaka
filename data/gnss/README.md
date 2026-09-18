# NinjaSat GNSS telemetry (public columns)

`ninjasat_gnss_2024-04_2024-11.csv`: one row per housekeeping packet carrying a GNSS fix,
2024-03-31 12:35 UTC to 2024-11-30 12:54 UTC, nominal cadence 600 s, 29,405 rows.
Provided by the NinjaSat team (RIKEN Pioneering Research Institute / RIKEN Nishina
Center). Rows are as downlinked; no quality control has been applied to this file.
`src/01_prepare_gps.py` applies the selection used in the paper.

## Columns released

| column | unit | meaning |
|---|---|---|
| `timestamp` | s (Unix) | packet time stamp |
| `gpsUtcTime` | s (Unix) | GNSS receiver UTC time of the fix |
| `gpsUtcTimeFraction` | s | sub-second part of `gpsUtcTime` |
| `gpsFixQual` | flag | NMEA fix quality; 1 = valid fix (the paper uses only these rows) |
| `gpsLatitude` | deg | geodetic latitude (WGS84) |
| `gpsLongitude` | deg | longitude |
| `gpsAltitudeMeters` | m | altitude above mean sea level as reported by the receiver (NMEA GGA) |
| `gpsNumOfSatsTracked` | count | GNSS satellites tracked |
| `gpsNumOfSatsInView` | count | GNSS satellites in view |
| `gpsGeoid` | m | geoid separation reported by the receiver; `gpsAltitudeMeters + gpsGeoid` is the ellipsoidal height used in the paper (Eq. 1) |
| `datetime` | UTC | `gpsUtcTime` as a readable time stamp |

## Columns not released

The same telemetry stream carries the following fields. They are part of the spacecraft
housekeeping data and are **not** included in this release. Their use requires explicit
permission from RIKEN; requests should go to the corresponding author
(Hayato Tanaka, tanaka.h.eff6@m.isct.ac.jp), who will forward them to the NinjaSat team.

| column | meaning | where the paper uses it |
|---|---|---|
| `gpsGroundSpeed` | receiver speed (km/h), treated as the ECEF speed in Eq. 1 | kinetic-energy term of $E_{J2}$; speed-window QC (26,000 to 29,000 km/h) |
| `gpsHDilutionOfPos` | horizontal dilution of precision | QC overview histogram only |
| `gpsIsPowerOn`, `gpsNumOfOvercurrentChecks`, `gpsNumOfOvercurrentsDetected`, `gpsCurrentMA` | receiver power and current | not used |

Consequences for reproduction:

- `src/01_prepare_gps.py` runs and skips the speed-window step with a printed note.
- `src/02_specific_energy.py` stops with an explanatory message. Every product from the
  3-hourly binning onwards (`data/processed/dEdt_3h.csv`, `data/figure_data/*.csv`) is
  supplied, so scripts 04 to 21 and 24 and all figure scripts run unchanged.
- The per-sample energy file (`specific_energy.csv`) is not supplied because, together
  with the released position columns, it would allow the speed to be recovered.
