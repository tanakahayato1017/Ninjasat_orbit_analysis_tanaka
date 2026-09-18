"""
Geomagnetic Activity Indices Extraction and Quiet Period Flagging

Purpose:
  Extract 3-hourly Kp and ap indices, and daily Ap values from GFZ official data files.
  Create quiet period flags (Kp < 3 and Kp < 4) for the analysis period.
  This supports reviewer request to demonstrate that results are robust against
  geomagnetic disturbances.

Input:
  - data/external/Kp_ap_since_1932.txt (3-hourly Kp/ap from GFZ)
  - data/external/Kp_ap_Ap_SN_F107_since_1932.txt (daily Ap/F10.7 from GFZ)

Output:
  - data/processed/geomag_3h.csv
    Columns: datetime, Kp, ap, quiet_kp_lt_3, quiet_kp_lt_4
  - data/processed/geomag_daily.csv
    Columns: datetime, Ap, F107_obs, F107_adj

Target Period: 2024-03-25 ~ 2024-12-05 (UTC)

TODO: #8 (Reviewer requirement: robustness against geomagnetic disturbances)

Author: Claude
Date: 2026-07-02
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
DATA_EXTERNAL = Path(__file__).parent.parent / "data" / "external"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
FILE_3H = DATA_EXTERNAL / "Kp_ap_since_1932.txt"
FILE_DAILY = DATA_EXTERNAL / "Kp_ap_Ap_SN_F107_since_1932.txt"

# Analysis period
START_DATE = datetime(2024, 3, 25, 0, 0, 0)  # UTC
END_DATE = datetime(2024, 12, 5, 23, 59, 59)  # UTC

print(f"Loading geomagnetic data from {FILE_3H} and {FILE_DAILY}")
print(f"Target period: {START_DATE} to {END_DATE} UTC")

# ============================================================================
# Parse 3-hourly Kp/ap file
# ============================================================================
# Format (from header):
#YYY MM DD hh.h hh._m        days      days_m     Kp   ap D
# Fixed format:
# YYYY MM DD hh.h hh._m days days_m Kp ap D
# Example:
# 1932 01 01 00.0 01.50     0.00000     0.06250  3.333   18 1

print("\n=== Parsing 3-hourly Kp/ap file ===")
df_3h_raw = pd.read_csv(
    FILE_3H,
    comment='#',
    sep=r'\s+',
    names=['YYYY', 'MM', 'DD', 'hh_start', 'hh_mid', 'days', 'days_m', 'Kp', 'ap', 'D'],
    dtype={
        'YYYY': int, 'MM': int, 'DD': int,
        'hh_start': float, 'hh_mid': float,
        'days': float, 'days_m': float,
        'Kp': float, 'ap': int, 'D': int
    }
)

print(f"Total 3h records loaded: {len(df_3h_raw)}")
print(f"Date range in file: {df_3h_raw['YYYY'].min()}-{df_3h_raw['MM'].min():02d}-{df_3h_raw['DD'].min():02d} "
      f"to {df_3h_raw['YYYY'].max()}-{df_3h_raw['MM'].max():02d}-{df_3h_raw['DD'].max():02d}")

# Convert to datetime (starting time of 3-hour interval)
df_3h_raw['datetime'] = pd.to_datetime(
    df_3h_raw[['YYYY', 'MM', 'DD']].astype(str).apply(lambda x: '-'.join(x), axis=1) +
    ' ' + df_3h_raw['hh_start'].apply(lambda x: f"{int(x):02d}:00:00"),
    format='%Y-%m-%d %H:%M:%S'
)

# Handle missing values (-1 for Kp and ap)
df_3h_raw['Kp'] = df_3h_raw['Kp'].replace(-1.0, np.nan)
df_3h_raw['ap'] = df_3h_raw['ap'].replace(-1, np.nan)

# Filter to target period
df_3h = df_3h_raw[(df_3h_raw['datetime'] >= START_DATE) &
                   (df_3h_raw['datetime'] <= END_DATE)].copy()

print(f"3h records in target period: {len(df_3h)}")
print(f"Period coverage: {df_3h['datetime'].min()} to {df_3h['datetime'].max()}")

# Create quiet period flags
# quiet_kp_lt_3: True if Kp < 3.0 (quieter)
# quiet_kp_lt_4: True if Kp < 4.0 (quiet)
df_3h['quiet_kp_lt_3'] = df_3h['Kp'] < 3.0
df_3h['quiet_kp_lt_4'] = df_3h['Kp'] < 4.0

# Select output columns
df_3h_output = df_3h[['datetime', 'Kp', 'ap', 'quiet_kp_lt_3', 'quiet_kp_lt_4']].copy()

# Save 3-hourly output
df_3h_output.to_csv(DATA_PROCESSED / "geomag_3h.csv", index=False)
print(f"\nSaved 3-hourly data to {DATA_PROCESSED / 'geomag_3h.csv'}")

# ============================================================================
# Parse daily Ap file
# ============================================================================
# Format (from header):
#YYY MM DD  days  days_m  Bsr dB    Kp1 ... Kp8  ap1 ... ap8    Ap  SN F10.7obs F10.7adj D
# We need: YYYY MM DD Ap F10.7obs F10.7adj

print("\n=== Parsing daily Ap file ===")
# Read with many columns; we'll select what we need
df_daily_raw = pd.read_csv(
    FILE_DAILY,
    comment='#',
    sep=r'\s+',
    # Column order from format definition: YYYY MM DD days days_m BSR dB Kp1-Kp8(8) ap1-ap8(8) Ap SN F10.7obs F10.7adj D
    names=['YYYY', 'MM', 'DD', 'days', 'days_m', 'BSR', 'dB',
           'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8',
           'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8',
           'Ap', 'SN', 'F107_obs', 'F107_adj', 'D'],
    dtype={
        'YYYY': int, 'MM': int, 'DD': int,
        'days': float, 'days_m': float,
        'BSR': int, 'dB': int,
        'Kp1': float, 'Kp2': float, 'Kp3': float, 'Kp4': float,
        'Kp5': float, 'Kp6': float, 'Kp7': float, 'Kp8': float,
        'ap1': int, 'ap2': int, 'ap3': int, 'ap4': int,
        'ap5': int, 'ap6': int, 'ap7': int, 'ap8': int,
        'Ap': int, 'SN': int, 'F107_obs': float, 'F107_adj': float, 'D': int
    }
)

print(f"Total daily records loaded: {len(df_daily_raw)}")
print(f"Date range in file: {df_daily_raw['YYYY'].min()}-{df_daily_raw['MM'].min():02d}-{df_daily_raw['DD'].min():02d} "
      f"to {df_daily_raw['YYYY'].max()}-{df_daily_raw['MM'].max():02d}-{df_daily_raw['DD'].max():02d}")

# Convert to datetime (00:00 UTC of the day)
df_daily_raw['datetime'] = pd.to_datetime(
    df_daily_raw[['YYYY', 'MM', 'DD']].astype(str).apply(lambda x: '-'.join(x), axis=1),
    format='%Y-%m-%d'
)

# Handle missing values
df_daily_raw['Ap'] = df_daily_raw['Ap'].replace(-1, np.nan)
df_daily_raw['F107_obs'] = df_daily_raw['F107_obs'].replace(-1.0, np.nan)
df_daily_raw['F107_adj'] = df_daily_raw['F107_adj'].replace(-1.0, np.nan)

# Filter to target period
df_daily = df_daily_raw[(df_daily_raw['datetime'] >= pd.Timestamp(START_DATE)) &
                         (df_daily_raw['datetime'] <= pd.Timestamp(END_DATE))].copy()

print(f"Daily records in target period: {len(df_daily)}")
print(f"Period coverage: {df_daily['datetime'].min()} to {df_daily['datetime'].max()}")

# Select output columns
df_daily_output = df_daily[['datetime', 'Ap', 'F107_obs', 'F107_adj']].copy()
df_daily_output.columns = ['datetime', 'Ap', 'F107_obs', 'F107_adj']

# Save daily output
df_daily_output.to_csv(DATA_PROCESSED / "geomag_daily.csv", index=False)
print(f"\nSaved daily data to {DATA_PROCESSED / 'geomag_daily.csv'}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n3-HOURLY DATA ({len(df_3h)} intervals):")
print(f"  Date range: {df_3h['datetime'].min()} to {df_3h['datetime'].max()}")

# Count quiet and disturbed intervals
n_lt_3 = df_3h['quiet_kp_lt_3'].sum()
n_gte_3 = (~df_3h['quiet_kp_lt_3']).sum()
n_lt_4 = df_3h['quiet_kp_lt_4'].sum()
n_gte_4 = (~df_3h['quiet_kp_lt_4']).sum()

print(f"\n  Kp < 3.0 (quieter periods):  {n_lt_3:4d} intervals ({100*n_lt_3/len(df_3h):5.1f}%)")
print(f"  Kp >= 3.0 (disturbed):       {n_gte_3:4d} intervals ({100*n_gte_3/len(df_3h):5.1f}%)")
print(f"  Kp < 4.0 (quiet periods):    {n_lt_4:4d} intervals ({100*n_lt_4/len(df_3h):5.1f}%)")
print(f"  Kp >= 4.0 (highly disturbed):{n_gte_4:4d} intervals ({100*n_gte_4/len(df_3h):5.1f}%)")

# Kp statistics
print(f"\n  Kp statistics:")
print(f"    Min:       {df_3h['Kp'].min():.2f}")
print(f"    Max:       {df_3h['Kp'].max():.2f}")
print(f"    Mean:      {df_3h['Kp'].mean():.2f}")
print(f"    Median:    {df_3h['Kp'].median():.2f}")
print(f"    Std Dev:   {df_3h['Kp'].std():.2f}")

# Find max Kp event
idx_max_kp = df_3h['Kp'].idxmax()
max_kp_val = df_3h.loc[idx_max_kp, 'Kp']
max_kp_time = df_3h.loc[idx_max_kp, 'datetime']
print(f"\n  Maximum Kp event:")
print(f"    Kp = {max_kp_val:.2f} on {max_kp_time}")

# Monthly breakdown
print(f"\n  Disturbed intervals (Kp >= 3.0) by month:")
df_3h['month'] = df_3h['datetime'].dt.to_period('M')
monthly_disturbed = df_3h.groupby('month')['quiet_kp_lt_3'].apply(lambda x: (~x).sum())
for month, count in monthly_disturbed.items():
    pct = 100 * count / df_3h[df_3h['month'] == month].shape[0]
    print(f"    {month}: {count:3d} intervals ({pct:5.1f}%)")

print(f"\nDAILY DATA ({len(df_daily)} days):")
print(f"  Date range: {df_daily['datetime'].min().date()} to {df_daily['datetime'].max().date()}")
print(f"  Ap statistics:")
print(f"    Min:       {df_daily['Ap'].min():4d} nT")
print(f"    Max:       {df_daily['Ap'].max():4d} nT")
print(f"    Mean:      {df_daily['Ap'].mean():6.1f} nT")
print(f"    Median:    {df_daily['Ap'].median():6.1f} nT")
print(f"  F10.7 obs (solar radio flux):")
print(f"    Min:       {df_daily['F107_obs'].min():7.1f} s.f.u.")
print(f"    Max:       {df_daily['F107_obs'].max():7.1f} s.f.u.")
print(f"    Mean:      {df_daily['F107_obs'].mean():7.1f} s.f.u.")

print("\n" + "="*70)
print("SUCCESS: All data processed and saved.")
print("="*70)
