"""
Oberaargletscher Volume Projection

This script projects the glacier volume of Oberaargletscher into the future based on
modeled mass balance data. It calculates the glacier's volume changes and estimates
the year of 50% volume loss and complete disappearance.

Inputs:
- Reconstructed mass balance CSV: `data/processed/reconstructed_mass_balance_1935_2024.csv`

Outputs:
- CSV: `data/processed/oberaarg_volume_projection.csv`
- Plot: `visualisations/oberaarg_volume_projection.png`

Assumptions:
- The reconstructed annual series was used as climatic forcing for a glacier-volume projection.
- Annual glacier mass change for each year was converted from metres water equivalent to ice volume change 
  using the glacier’s 2024 HY area and an assumed ice density of 900 kg/m³.
- Glacier volume at the end of 2024 HY was estimated by applying accumulated mass changes to an externally 
  supplied reference volume for 2020 (Oberaargletscher, Goodbye Glaciers?!, 2025).
- For forward projections, the long-term mean mass-balance rate over the past 30 years was assumed to remain constant.
- In a baseline scenario, the glacier area was held fixed at its 2024 HY value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- USER PARAMETERS ----------------
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB_CSV = os.path.join(BASE_DIR, "data/processed/reconstructed_mass_balance_1931_2024.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data/processed/oberaarg_volume_projection.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "visualisations/projection/oberaarg_volume_projection.png")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# Column names expected in CSV
YEAR_COL = "hydro_year"
MB_COL = "modeled_mb"

# Glacier properties
V_2020_KM3 = 0.14  # Ice volume in 2020 [km³] (given)
AREA_KM2 = 3.26    # Glacier area (assumed constant) [km²]
RHO_ICE = 900.0    # Ice density [kg/m³]
RHO_WATER = 1000.0 # Water density [kg/m³]

# Projection control
START_PROJ_YEAR = 2025  # First projected year (we already applied up to 2024)
STOP_YEAR = 2060        # Maximum year to project

# ------------------- LOAD MB CSV -------------------
df = pd.read_csv(MB_CSV)

# Check columns
if YEAR_COL not in df.columns or MB_COL not in df.columns:
    print("Columns in CSV:", df.columns.tolist())
    raise SystemExit(f"Expected columns '{YEAR_COL}' and '{MB_COL}' in {MB_CSV}")

# Ensure years are integers and data is sorted
df[YEAR_COL] = df[YEAR_COL].astype(int)
df = df.sort_values(YEAR_COL).reset_index(drop=True)

# -------------- Compute Average MB (1994-2024) -----------
mask_1994_2024 = (df[YEAR_COL] >= 1994) & (df[YEAR_COL] <= 2024)
if mask_1994_2024.sum() == 0:
    raise SystemExit("No MB values found in 1994-2024 range in CSV. Check file.")
avg_mb_1994_2024 = df.loc[mask_1994_2024, MB_COL].mean()  # Average MB [m w.e./year]
print(f"Average MB (1994-2024): {avg_mb_1994_2024:.3f} m w.e./year")

# ---------- Compute Volume Change (2020 -> 2024) -----------
mask_2020_2024 = (df[YEAR_COL] >= 2020) & (df[YEAR_COL] <= 2024)
mb_2020_2024 = df.loc[mask_2020_2024, MB_COL].values
sum_mb_2020_2024 = np.sum(mb_2020_2024)

# Conversions
A_M2 = AREA_KM2 * 1e6  # Glacier area [m²]
V_2020_M3 = V_2020_KM3 * 1e9  # Ice volume in 2020 [m³]

# Water equivalent volume change
dV_water_m3 = sum_mb_2020_2024 * A_M2
# Convert to ice volume change [m³ ice]
dV_ice_m3 = dV_water_m3 * (RHO_WATER / RHO_ICE)

V_2024_M3 = V_2020_M3 + dV_ice_m3
V_2024_KM3 = V_2024_M3 / 1e9

print(f"Volume change 2020-2024 (water eq) [m³]: {dV_water_m3:,.0f}")
print(f"Volume change 2020-2024 (ice) [m³]: {dV_ice_m3:,.0f}")
print(f"Estimated glacier volume at end of 2024: {V_2024_KM3:.4f} km³")

# ---------------------- Project Future Volume ----------------------
years = list(range(START_PROJ_YEAR, STOP_YEAR + 1))
V = [V_2024_M3]

for year in years:
    # Apply average mass balance (m w.e.) per year
    dVw = avg_mb_1994_2024 * A_M2  # Water equivalent volume change [m³/year]
    dVi = dVw * (RHO_WATER / RHO_ICE)  # Ice volume change [m³/year]
    V_new = V[-1] + dVi
    if V_new <= 0:  # Stop projection if volume reaches zero or below
        V.append(0)
        break
    V.append(V_new)

V = np.array(V)  # Convert to NumPy array
years_all = [2024] + years[:len(V) - 1]  # Truncate years to match the volume array
V_KM3 = V / 1e9  # Convert volume to km³

# ---------------------- Save Projection Results ----------------------
out_df = pd.DataFrame({"Year": years_all, "Volume_km3": V_KM3})
out_df.to_csv(OUTPUT_CSV, index=False)

# ---------------------- Analyze Key Milestones ----------------------
# Year of 50% volume loss relative to 2020
half_volume = V_2020_M3 * 0.5
year_half = next((y for y, vol in zip(years_all, V) if vol <= half_volume), None)

if year_half:
    print(f"Year of 50% volume (or first year ≤50%): {year_half}")
else:
    print(f"No 50% volume loss within projection window (up to {STOP_YEAR}).")

# Year of complete disappearance (volume ≤ 0)
year_zero = next((y for y, vol in zip(years_all, V) if vol <= 0), None)

if year_zero:
    print(f"Glacier disappears in year: {year_zero}")
else:
    print(f"Glacier does not fully disappear within projection window (up to {STOP_YEAR}).")

# ---------------------- Plot Volume Projection ----------------------
plt.figure(figsize=(7, 4))
plt.plot(years_all, V_KM3, '-o', label="Projected Volume")
plt.axhline(V_2020_KM3, color='k', linestyle='--', label='V_2020')
plt.axhline(half_volume / 1e9, color='r', linestyle=':', label='50% V_2020')
plt.xlabel('Year')
plt.ylabel('Glacier Volume (km³)')
plt.title('Oberaargletscher Volume Projection (Constant Area)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=200)
plt.close()
print("Saved projection plot to:", OUTPUT_PLOT)
print("Saved projection table to:", OUTPUT_CSV)
