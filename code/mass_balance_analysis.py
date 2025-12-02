"""
Mass Balance Analysis for Oberaargletscher

This script generates four graphs based on the mass balance data provided in the CSV files:
1. Cumulative mass balance over years with uncertainties.
2. Annual mass balance with anomalies and geodetic balance comparison.
3. Glaciological mass balance and adjusted calibrated glaciological mass balance over whole observation period 2013-2024.

Data Sources:
- `data/raw/Calibration.csv`: Contains columns for Year, Glaciological, Geodetic, and Geodetic Calibrated balances.
- `data/raw/Calibration_whole period.csv`: Contains glaciological mass balance for the entire period.

Outputs:
- Saves the generated plots as PNG files in `visualisations/Analysis`.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- Constants and File Paths ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_FILE = os.path.join(BASE_DIR, "data/raw/Calibration.csv")
CALIBRATION_WHOLE_PERIOD_FILE = os.path.join(BASE_DIR, "data/raw/Calibration_whole period.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualisations", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Helper Functions ----------------------
def calculate_cumulative_uncertainty(data, start_year, annual_uncertainty):
    """
    Calculate cumulative uncertainty for each year based on annual uncertainty.
    """
    data['Cumulative_Uncertainty'] = annual_uncertainty * np.sqrt(data['Year'] - start_year)
    return data

def plot_cumulative_mass_balance(data, output_file):
    """
    Plot cumulative mass balance with uncertainties for Glaciological, Geodetic, and Geodetic Calibrated balances.
    """
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=0.8)  # Anchor x-axis to 0

    # Glaciological balance with uncertainty
    plt.plot(data['Year'], data['Glaciological'].cumsum(), label='Glaciological MB', color='blue')
    plt.fill_between(
        data['Year'],
        data['Glaciological'].cumsum() - data['Cumulative_Uncertainty'],
        data['Glaciological'].cumsum() + data['Cumulative_Uncertainty'],
        color='blue', alpha=0.2, label='Glaciological Uncertainty'
    )

    # Geodetic balance with uncertainty
    geodetic_uncertainty = np.where(data['Year'] == 2013, 0, 0.61)  # Set uncertainty to 0 for 2013
    plt.plot(data['Year'], data['Geodetic'].cumsum(), label='Geodetic MB', color='orange')
    plt.fill_between(
        data['Year'],
        data['Geodetic'].cumsum() - geodetic_uncertainty,
        data['Geodetic'].cumsum() + geodetic_uncertainty,
        color='orange', alpha=0.2, label='Geodetic Uncertainty'
    )

    # Geodetic calibrated balance
    plt.plot(data['Year'], data['Geodetic_cal'].cumsum(), label='Calibrated MB', color='green', linestyle='--')

    # Plot settings
    plt.xlabel('Hydrological Year (1 Oct - 30 Sep)')
    plt.ylabel('Cumulative Mass Balance (m w.e.)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_annual_mass_balance(data, mean_glaciological, output_file):
    """
    Plot annual mass balance with anomalies and geodetic balance comparison.
    """
    anomalies = data['Glaciological'] - mean_glaciological

    plt.figure(figsize=(12, 8))
    plt.axhline(0, color='black', linewidth=0.8)  # Anchor x-axis to 0

    # Bars for glaciological balance
    plt.bar(data['Year'], data['Glaciological'], color='lightblue', label='Glaciological MB')

    # Line for mean glaciological balance
    plt.axhline(mean_glaciological, color='blue', linestyle='--', label=f'Mean Glaciological MB')

    # Anomalies with arrows
    for i, year in enumerate(data['Year']):
        if anomalies.iloc[i] > 0:
            plt.arrow(year, mean_glaciological, 0, anomalies.iloc[i], color='green', width=0.03, head_width=0.06, head_length=0.03, length_includes_head=True)
        else:
            plt.arrow(year, mean_glaciological, 0, anomalies.iloc[i], color='red', width=0.03, head_width=0.06, head_length=0.03, length_includes_head=True)

    # Line for geodetic balance
    plt.plot(data['Year'], data['Geodetic'], color='orange', label='Geodetic MB')

    # Plot settings
    plt.xlabel('Hydrological Year (1 Oct - 30 Sep)')
    plt.ylabel('Annual Mass Balance (m w.e.)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_adjusted_mass_balance(data, adjustment_constant, output_file):
    """
    Plot glaciological mass balance and adjusted calibrated glaciological mass balance.
    """
    data['Adjusted_Glaciological'] = data['Glaciological'] + adjustment_constant

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=0.8)  # Anchor x-axis to 0

    # Glaciological balance
    plt.plot(data['Year'], data['Glaciological'], label='Glaciological MB', color='blue')

    # Adjusted glaciological balance
    plt.plot(data['Year'], data['Adjusted_Glaciological'], label='Calibrated MB', color='green', linestyle='--')

    # Plot settings
    plt.xlabel('Hydrological Year (1 Oct - 30 Sep)')
    plt.ylabel('Mass Balance (m w.e.)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# ---------------------- Main Script ----------------------
if __name__ == "__main__":
    # Load calibration data
    calibration_data = pd.read_csv(CALIBRATION_FILE)
    calibration_data.columns = ['Year', 'Glaciological', 'Geodetic', 'Geodetic_cal']

    # Filter data for cumulative plot (starting from 2013)
    cumulative_data = calibration_data[calibration_data['Year'] >= 2013]
    cumulative_data = calculate_cumulative_uncertainty(cumulative_data, start_year=2013, annual_uncertainty=0.36)

    # Plot 1: Cumulative mass balance
    plot_cumulative_mass_balance(cumulative_data, output_file=os.path.join(OUTPUT_DIR, "cumulative_mass_balance.png"))

    # Filter data for annual plot (starting from 2014)
    annual_data = calibration_data[calibration_data['Year'] >= 2014]
    mean_glaciological = annual_data['Glaciological'].mean()

    # Plot 2: Annual mass balance with anomalies
    plot_annual_mass_balance(annual_data, mean_glaciological, output_file=os.path.join(OUTPUT_DIR, "annual_mass_balance.png"))

    # Load whole-period calibration data
    whole_period_data = pd.read_csv(CALIBRATION_WHOLE_PERIOD_FILE)
    whole_period_data.columns = ['Year', 'Glaciological']

    # Calculate adjustment constant
    adjustment_constant = calibration_data['Geodetic'].mean() - calibration_data['Glaciological'].mean()

    # Plot 3: Adjusted mass balance
    plot_adjusted_mass_balance(whole_period_data, adjustment_constant, output_file=os.path.join(OUTPUT_DIR, "adjusted_glaciological_mass_balance.png"))