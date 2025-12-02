#!/usr/bin/env python3
"""
Calibrate Degree Day Factor (DDF) and reconstruct annual mass balance
for Oberaargletscher using Grimsel (GRH) monthly homogenized data.

Inputs:
- Monthly homogenized file (MeteoSwiss): `data/raw/climate-reports-tables-homogenized_GRH.txt`
- Observed mass balance: `data/raw/Calibration_whole period.csv`

Assumptions:
- Precipitation can be modeled with a "block rain model."
- Snow and rain can be distinguished through a threshold temperature of T0 = 2.0 °C.
- Temperature at any elevation can be computed using a lapse rate of 6.5 °C/km.
- Glacier average elevation = 2850 m a.s.l.
- Melt occurs for temperatures above 0 °C.
- Days per month = 365/12 ≈ 30.4167.
- Hydrological year = 1 Oct - 30 Sep (labeled by the October year, as in observed data).

Outputs:
- Saves CSV: `data/processed/reconstructed_mass_balance_1935_2024.csv`
- Saves figures: mass balance timeseries, PDD vs observed MB scatter, and modeled MB bar plot in `visualisations/`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

# ---------------------- Constants and File Paths -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METEO_FILE = os.path.join(BASE_DIR, "data/raw/climate-reports-tables-homogenized_GRH.txt")
OBSERVED_MB_FILE = os.path.join(BASE_DIR, "data/raw/Calibration_whole period.csv")
OUTDIR = os.path.join(BASE_DIR, "visualisations", "calibration")
os.makedirs(OUTDIR, exist_ok=True)

# Station metadata (Grimsel Hospiz)
STATION_ELEV = 1980.0  # m a.s.l.

# Glacier elevation (assumed)
GLACIER_ELEV = 2850.0  # m a.s.l.

# Physics / model constants
T0 = 2.0  # °C threshold for snow/rain
TEMP_LAPSE_RATE = 6.5  # °C per km
DAYS_PER_MONTH = 365.0 / 12.0  # ≈ 30.4167 days
T_MELT = 0.0  # melt occurs for T > 0

# ---------------------- Helper Functions -------------------------
def load_observed_mb(file_path):
    """
    Load observed mass balance data from a CSV file.
    Returns a dictionary with years as keys and mass balance as values.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Strip whitespace and clean column names
    df.columns = df.columns.str.strip()
    
    # Check if the required columns exist
    if "Year" not in df.columns or "Glaciological" not in df.columns:
        raise ValueError(f"Expected columns 'Year' and 'Glaciological' in {file_path}. Found columns: {df.columns.tolist()}")
    
    # Create a dictionary from the DataFrame
    observed_mb = dict(zip(df["Year"], df["Glaciological"]))
    return observed_mb

def read_meteo_monthly_grh(path):
    """
    Read MeteoSwiss homogenized file and return a DataFrame with columns: Year, Month, Temperature, Precipitation.
    """
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("latin1")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    lines = text.splitlines()
    # Find the first line of data (where the year column starts)
    first_data_idx = next(
        (i for i, line in enumerate(lines) if line.strip() and line.split()[0].isdigit()), None
    )
    if first_data_idx is None:
        raise RuntimeError("Could not find numeric start in meteo file. Check file format.")

    # Read the data into a DataFrame
    df = pd.read_csv(
        path,
        encoding="latin1",
        sep=r"\s+",
        skiprows=first_data_idx,
        names=["Year", "Month", "Temperature", "Precipitation"],
        usecols=[0, 1, 2, 3],
        na_values=["NA"],
    )
    # Strip whitespace from column names (if any)
    df.columns = df.columns.str.strip()
    # Drop rows with missing Year or Month values
    df = df.dropna(subset=["Year", "Month"])
    # Ensure Year and Month are integers
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    return df

def add_hydrological_year(df):
    """
    Assign hydrological year (labeled by October year) to each row.
    """
    df = df.copy()
    df["hydro_year"] = df["Year"].where(df["Month"] >= 10, df["Year"] - 1)
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=15))
    return df

def apply_lapse_and_partition(df):
    """
    Adjust temperature to glacier elevation, compute PDD, and partition precipitation into snow/rain.
    """
    df = df.copy()
    elev_diff_km = (GLACIER_ELEV - STATION_ELEV) / 1000.0
    df["T_glac"] = df["Temperature"] - TEMP_LAPSE_RATE * elev_diff_km
    df["PDD_month"] = np.maximum(0.0, df["T_glac"].fillna(0.0)) * DAYS_PER_MONTH
    df["Snow_mm"] = np.where(df["T_glac"] < T0, df["Precipitation"].fillna(0.0), 0.0)
    df["Rain_mm"] = np.where(df["T_glac"] >= T0, df["Precipitation"].fillna(0.0), 0.0)
    df["Snow_mwe"] = df["Snow_mm"] / 1000.0
    return df

def compute_annual_mb(df_monthly, ddf):
    """
    Compute annual mass balance (m w.e.) per hydrological year using the given DDF.
    """
    df = df_monthly.copy()
    df["Ablation_mwe"] = ddf * df["PDD_month"]
    df["MB_month_mwe"] = df["Snow_mwe"] - df["Ablation_mwe"]
    return df.groupby("hydro_year")["MB_month_mwe"].sum()

def calibration_objective(ddf, df_monthly, obs_dict):
    """
    Objective function for DDF calibration: minimize squared differences between modeled and observed MB.
    """
    modeled = compute_annual_mb(df_monthly, ddf)
    obs_years = sorted(obs_dict.keys())
    modeled_vals = modeled.reindex(obs_years)
    obs_vals = np.array([obs_dict[y] for y in obs_years], dtype=float)
    if modeled_vals.isna().any():
        return 1e6 + np.nansum((modeled_vals.fillna(0.0) - obs_vals) ** 2)
    return np.mean((modeled_vals.values - obs_vals) ** 2)

# ---------------------- Main Script -------------------------
def main():
    print("Loading observed mass balance data...")
    observed_mb = load_observed_mb(OBSERVED_MB_FILE)

    print("Reading meteorological file:", METEO_FILE)
    df = read_meteo_monthly_grh(METEO_FILE)
    df = add_hydrological_year(df)
    df = apply_lapse_and_partition(df)


    bounds = (0.0005, 0.02)
    res = minimize_scalar(
        lambda x: calibration_objective(x, df, observed_mb),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    ddf_cal = res.x
    print(f"\nCalibrated DDF = {ddf_cal:.6f} m w.e. °C^-1 day^-1")
    print(f"Calibration success: {res.success}, objective={res.fun:.6e}")

    annual_modeled = compute_annual_mb(df, ddf_cal)
    annual_pdd = df.groupby("hydro_year")["PDD_month"].sum()
    obs_series = pd.Series(observed_mb)

    df_out = pd.DataFrame({
        "modeled_mb": annual_modeled,
        "annual_pdd": annual_pdd,
        "observed_mb": obs_series.reindex(annual_modeled.index),
    }).sort_index()

    csv_out = os.path.join(BASE_DIR, "data/processed/reconstructed_mass_balance_1931_2024.csv")
    df_out.to_csv(csv_out, float_format="%.4f")
    print("Saved reconstructed annual mass-balance CSV to:", csv_out)

    # Save figures in the `visualisations` folder
    ts_png = os.path.join(OUTDIR, "modeled_vs_observed_mb_timeseries.png")

    # Generate and save the time series plot
    plt.figure(figsize=(9, 4))
    years = df_out.index.values

    # Plot modeled and observed MB
    plt.plot(years, df_out["modeled_mb"], "-o", label="Modeled MB (DDF calibrated)")
    plt.plot(years, df_out["observed_mb"], "s", label="Observed MB")

    # Calculate and plot the mean modeled MB
    mean_modeled_mb = df_out["modeled_mb"].mean()
    plt.axhline(mean_modeled_mb, color="red", linestyle="--", label=f"Mean Modeled MB ({mean_modeled_mb:.2f} m w.e.)")

    # Add plot settings
    plt.axhline(0, color="k", lw=0.6)
    plt.xlabel("Hydrological Year")
    plt.ylabel("Mass Balance (m w.e.)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(ts_png, dpi=200)
    plt.close()
    print("Saved timeseries plot to:", ts_png)

    # Save scatter plot in the `visualisations` folder
    scatter_png = os.path.join(OUTDIR, "pdd_vs_observed_mb_scatter.png")

    # Generate and save the scatter plot
    plt.figure(figsize=(6, 5))
    common_years = sorted(observed_mb.keys())
    pdd_obs = annual_pdd.reindex(common_years)
    obsvals = pd.Series(observed_mb).reindex(common_years)

    # Scatter plot of PDD vs Observed MB
    plt.scatter(pdd_obs.values, obsvals.values, c="C0", s=50, label="Observed Data")

    # Linear fit for visualization
    mask = ~np.isnan(pdd_obs.values) & ~np.isnan(obsvals.values)
    if mask.sum() >= 2:
        m, c = np.polyfit(pdd_obs.values[mask], obsvals.values[mask], 1)
        xx = np.linspace(pdd_obs.min() * 0.9, pdd_obs.max() * 1.1, 50)
        plt.plot(xx, m * xx + c, "--", color="gray", label=f"Linear Fit: y = {m:.3f}x + {c:.3f}")

    # Add plot settings
    plt.xlabel("Annual PDD (°C·days)")
    plt.ylabel("Observed Annual MB (m w.e.)")
    plt.title("Annual PDD vs Observed MB")
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=200)
    plt.close()
    print("Saved PDD vs Observed MB scatter plot to:", scatter_png)

if __name__ == "__main__":
    main()