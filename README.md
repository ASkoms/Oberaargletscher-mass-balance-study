# Oberaargletscher Mass Balance Study

## Overview
This repository contains code, data, and analysis for the geodetic and glaciological mass balance study of Oberaargletscher, Swiss Alps. The study combines field measurements, DEM differencing, and temperature-index modeling to assess recent mass loss and project future glacier volume.

## Repository Structure
The repository is organized as follows:

```
LICENCE
README.md
code/
    ddf_calibration.py          # Script to calibrate Degree Day Factor (DDF) and reconstruct mass balance
    mass_balance_analysis.py    # Script to analyze and visualize mass balance data
    volume_projection.py        # Script to project glacier volume into the future
    requirements.txt            # Python dependencies
data/
    raw/
        Calibration.csv                     # Glaciological and geodetic mass balance data
        Calibration_whole period.csv        # Glaciological mass balance for the entire observation period (2013-2024 HY)
        climate-reports-tables-homogenized_GRH.txt  # Monthly homogenized meteorological data from Grimsel Hospiz (Homogeneous data series since 1864 - MeteoSwiss (2025). Available at: https://www.meteoswiss.admin.ch/services-and-publications/applications/ext/climate-tables-homogenized.html (Accessed: 24 November 2025).)
    processed
        reconstructed_mass_balance_1931_2024.csv    # Reconstructed mass balance data
        oberaarg_volume_projection.csv             # Projected glacier volume data
visualisations/
    analysis/
        cumulative_mass_balance.png                # Cumulative mass balance plot
        annual_mass_balance.png                    # Annual mass balance plot
        adjusted_glaciological_mass_balance.png    # Adjusted glaciological mass balance plot
    calibration/
        modeled_vs_observed_mb_timeseries.png      # Modeled vs observed mass balance time series
        pdd_vs_observed_mb_scatter.png             # PDD vs observed mass balance scatter plot
    projection/
        oberaarg_volume_projection.png             # Glacier volume projection plot
```

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required Python packages using the following command:
```bash
pip install -r code/requirements.txt
```

### Running the Scripts
1. **Analyze Mass Balance:**
   - Run `mass_balance_analysis.py` to generate visualizations of cumulative, annual, and adjusted mass balance.
   - Outputs:
     - `visualisations/analysis/cumulative_mass_balance.png`
     - `visualisations/analysis/annual_mass_balance.png`
     - `visualisations/analysis/adjusted_glaciological_mass_balance.png`

   ```bash
   python code/mass_balance_analysis.py
   ```

2. **Calibrate Degree Day Factor (DDF):**
   - Run `ddf_calibration.py` to calibrate the DDF and reconstruct annual mass balance.
   - Outputs:
     - `data/processed/reconstructed_mass_balance_1935_2024.csv`
     - `visualisations/calibration/modeled_vs_observed_mb_timeseries.png`
     - `visualisations/calibration/pdd_vs_observed_mb_scatter.png`

   ```bash
   python code/ddf_calibration.py
   ```

3. **Project Glacier Volume:**
   - Run `volume_projection.py` to project the glacier volume into the future.
   - Outputs:
     - `data/processed/oberaarg_volume_projection.csv`
     - `visualisations/projection/oberaarg_volume_projection.png`

   ```bash
   python code/volume_projection.py
   ```

## Expected Outputs
- **Reconstructed Mass Balance Data:** A CSV file containing modeled and observed mass balance data for the period 1935–2024.
- **Visualizations:**
  - Cumulative, annual, and adjusted mass balance plots.
  - Modeled vs observed mass balance time series.
  - Scatter plot of Positive Degree Days (PDD) vs observed mass balance.
  - Glacier volume projection plot.
- **Projected Glacier Volume Data:** A CSV file containing the projected glacier volume for the years 2024–2060.

## Authors
- Andrejs Skomorohovs

## License
This project is licensed under the MIT License. See the [LICENSE](LICENCE) file for details.