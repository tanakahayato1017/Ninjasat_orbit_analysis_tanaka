# Original-submission material (superseded)

These files were published with the first version of the manuscript (April 2026). That
version used the difference between the SGP4-propagated TLE altitude and the GNSS
altitude as the drag proxy. The revised paper replaces it with the orbital-energy decay
rate $-dE_{J2}/dt$ computed directly from the GNSS fixes, and the wavelength-split
response times reported in the original version were found not to be statistically
resolved. Nothing here is used by the revised paper's figures or numbers. The files are
kept so that the reviewer correspondence and the audits in `src/04, 08, 12, 13, 20, 21`
remain reproducible.

| directory | content |
|---|---|
| `altitude_diff_monthly_ext/` | monthly TLE-minus-GNSS altitude difference (2-orbit bins), 2024-04 to 2024-11: the legacy proxy input to the audit scripts |
| `GNSS_data/` | the three monthly GNSS extracts (2024-07 to 09) published originally; superseded by `data/gnss/` (same columns, full period) |
| `comparison_GNSS_TLE/` | per-fix GNSS vs SGP4 position comparison, 2024-07 to 09 |
| `Savitzky-golay/` | smoothed altitude-rate series used by the original method |
| `correlation/` | original per-channel lag-correlation tables, 2024-07 to 09 |
| `example_altitude_fitting/` | one-day example of the original altitude fit |
