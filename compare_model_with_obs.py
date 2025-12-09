
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rex import MultiFileResourceX, ResourceX
import xarray as xr
from matplotlib.patches import Patch

### Define paths
supercc_datapath = '/datasets/sup3rcc/conus_ecearth3cc_ssp245_r1i1p1f1/v0.2.2/' 
obs_datapath = 'Data/'

### Read met station data (1 Jan 2020-1 Jan 2025)
obs_df = pd.read_csv(obs_datapath + 'GHCNd_met_station_orlando_intl.csv') 



### Met station data (from https://www.ncei.noaa.gov/cdo-web/search) 
# AWND - Average daily wind speed (mph)
# PGTM - Peak gust time (HHMM)
# TAVG - Average daily temperature (C)
# TMAX - Maximum daily temperature (C)
# TMIN - Minimum daily temperature (C)
# WDF2 - Direction of fastest 2-minute wind speed (degrees)
# WDF5 - Direction of fastest 5-second wind speed (degrees)
# WSF2 - Fastest 2-minute wind speed (mph)
# WSF5 - Fastest 5-second wind speed (mph)
# WT01 - Fog, ice fog, or freezing fog (may include heavy fog)
# WT02 - Heavy fog or heaving freezing fog (not always distinguished from fog)
# WT03 - Thunder
# WT08 - Smoke or haze



### Sup3r data
## Read Sup3r data for Orlando International Airport for 2020-2024
# Read only one file
#fn_2020 = 'sup3rcc_conus_ecearth3cc_ssp245_r1i1p1f1_trh_2020.h5'
#ds = xr.open_dataset(os.path.join(supercc_datapath, fn_2020), engine='rex')
# relativehumidity_2m
# temperature_2m



## To open multiple files
# Open files (2015 through 2059 available)
files_super = os.path.join(supercc_datapath, 'sup3rcc_conus_ecearth3cc_ssp245_r1i1p1f1_trh_20*.h5')
ds_super_all = xr.open_mfdataset(files_super, engine='rex', compat='override', coords='minimal')

## Extract at location
# Coord for Orlando Intl Airport
lat0 = 28.4294
lon0 = -81.3089

# compute distance to each point (lazy Dask array)
dist = ((ds_super_all.latitude - lat0)**2 + (ds_super_all.longitude - lon0)**2)

# get index of the minimum distance
target_gid = dist.argmin(dim='gid').compute()

# select timeseries data with the target_gid
ts = ds_super_all.sel(gid=target_gid)
rh_ts = ts['relativehumidity_2m']     # RH time series
temp_ts = ts['temperature_2m']        # Temp time series
df = ts[['relativehumidity_2m', 'temperature_2m']].to_dataframe()



### Calculate the max daily temp for the model data and save to NetCDF (if not already saved) 
if not os.path.exists("Data/temp_daily_max.nc"):
    ## Calculate max daily temp of all years from files_super
    temp_daily_max = ts['temperature_2m'].resample(time='1D').max()
    temp_daily_max_float = temp_daily_max.compute().astype("float32") # Cannot save dtype int16 to netcdf

    ## Save to NetCDF
    ds = xr.Dataset({"temperature_2m": temp_daily_max_float})
    os.makedirs("Data", exist_ok=True)
    # Prepare data
    data = temp_daily_max.values.astype("float32")
    time = temp_daily_max["time"].values
    # Handle scalar coordinates robustly
    attrs = {}
    for k in temp_daily_max.coords:
        coord = temp_daily_max.coords[k]
        if coord.shape == ():  # scalar
            val = coord.values
            # Decode bytes if needed
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            attrs[k] = val
    # Create clean Dataset
    ds_clean = xr.Dataset(
        {"temperature_2m": ("time", data)},
        coords={"time": time},
        attrs=attrs
    )
    # Save to NetCDF
    ds_clean.to_netcdf("Data/temp_daily_max.nc", engine="netcdf4", format="NETCDF4")
else:
    # Read from file 
    ds = xr.open_dataset('Data/temp_daily_max.nc')
    temp_daily_max = ds['temperature_2m']


""" handler_2020 = MultiFileResourceX(os.path.join(supercc_datapath, fn_2020))
# Coord for Orlando Intl Airport
coord = (28.4294, -81.3089)  # (lat, lon)

dsets = handler_2020.lat_lon_grid(coord)
meta = handler_2020.meta
ti_2020 = handler_2020.time_index
data_2020 = pd.DataFrame({}) """



### Plot one year of max temp data and label how many days exceed threshold in each obs and model data
def plot_daily_max_year_with_threshold(temp_daily_max, obs_df, year, threshold=30):
    # --- Observations ---
    obs_df = obs_df.copy()
    obs_df['DATE'] = pd.to_datetime(obs_df['DATE'])
    obs_year = obs_df[obs_df['DATE'].dt.year == year].set_index('DATE')
    obs_tmax = (obs_year['TMAX'] - 32) * 5/9  # convert to °C

    # --- Model (xarray) ---
    model_year = temp_daily_max.sel(time=str(year))
    model_tmax = model_year.to_pandas().squeeze()

    # --- Compute day of year for plotting ---
    obs_x = obs_tmax.index.dayofyear
    model_x = model_tmax.index.dayofyear

    # --- Count days above threshold ---
    obs_days_above = (obs_tmax > threshold).sum()
    model_days_above = (model_tmax > threshold).sum()
    diff_days = obs_days_above - model_days_above  # difference

    # --- Create month ticks ---
    months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='MS')
    month_ticks = months.dayofyear
    month_labels = months.month

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12,5))

    # Shade summer months (JJA)
    jja_start = pd.Timestamp(f'{year}-06-01').dayofyear
    jja_end = pd.Timestamp(f'{year}-08-31').dayofyear
    ax.axvspan(jja_start, jja_end, color='yellow', alpha=0.1, label='JJA')

    # Plot model and observed
    ax.plot(model_x, model_tmax.values, label='Model Daily Max Temp')
    ax.plot(obs_x, obs_tmax.values, label='Observed TMAX (°C)')

    # Shade days above threshold
    ax.fill_between(model_x, model_tmax.values, threshold, where=(model_tmax.values>threshold), 
                    color='red', alpha=0.2, label=f'Model >{threshold}°C')
    ax.fill_between(obs_x, obs_tmax.values, threshold, where=(obs_tmax.values>threshold), 
                    color='blue', alpha=0.2, label=f'Obs >{threshold}°C')

    # Horizontal line at threshold
    ax.axhline(threshold, linestyle='--', linewidth=1, color='k', label=f'{threshold}°C')

    # --- Annotate number of days above threshold in upper-left with background box ---
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')

    ax.text(0.01, 0.96, f'Obs days > {threshold}°C: {obs_days_above}', 
            transform=ax.transAxes, color='blue', fontsize=10, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.91, f'Model days > {threshold}°C: {model_days_above}', 
            transform=ax.transAxes, color='red', fontsize=10, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.86, f'Obs - Model days > {threshold}°C: {diff_days}', 
            transform=ax.transAxes, color='black', fontsize=10, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)

    # Labels, ticks
    ax.set_title(f"Daily Max Temperature – {year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'Figures/daily_max_temp_{year}_with_threshold.png')
    plt.show()


### Plot bar chart of number of days obs and model are above temp threshold for a span of years
def plot_obs_minus_model_days_complete_years_bar_colored_legend(temp_daily_max, obs_df, threshold=30):
    # Ensure DATE is datetime
    obs_df = obs_df.copy()
    obs_df['DATE'] = pd.to_datetime(obs_df['DATE'])

    # Determine all unique years in obs_df
    years = sorted(obs_df['DATE'].dt.year.unique())

    diff_days_list = []
    complete_years = []

    for year in years:
        # --- Observations ---
        obs_year = obs_df[obs_df['DATE'].dt.year == year].set_index('DATE')
        # Skip year if observation data is incomplete
        if len(obs_year) < 365:
            continue
        obs_tmax = (obs_year['TMAX'] - 32) * 5/9  # °C
        obs_days_above = (obs_tmax > threshold).sum()

        # --- Model ---
        try:
            model_year = temp_daily_max.sel(time=str(year))
            model_tmax = model_year.to_pandas().squeeze()
            # Skip year if model data is incomplete
            if len(model_tmax) < 365:
                continue
            model_days_above = (model_tmax > threshold).sum()
        except KeyError:
            continue

        # --- Difference ---
        diff_days = obs_days_above - model_days_above
        diff_days_list.append(diff_days)
        complete_years.append(year)

    # --- Create DataFrame ---
    df_diff = pd.DataFrame({
        'Year': complete_years,
        'Obs - Model days > threshold': diff_days_list
    })

    # --- Plot bar chart with coloring ---
    colors = ['green' if x > 0 else 'red' for x in df_diff['Obs - Model days > threshold']]

    plt.figure(figsize=(10,5))
    bars = plt.bar(df_diff['Year'], df_diff['Obs - Model days > threshold'], color=colors)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    # Annotate each bar with value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 
                 height + np.sign(height)*0.5, 
                 f'{int(height)}', ha='center', va='bottom' if height>0 else 'top', fontsize=9)

    # --- Legend ---
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='Obs > Model'),
        Patch(facecolor='red', edgecolor='k', label='Obs < Model')
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel("Year")
    plt.ylabel(f"Obs - Model days > {threshold}°C")
    plt.title(f"Difference in Number of Days Above {threshold}°C (Obs - Model) – Complete Years Only")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('Figures/obs_minus_model_days_above_threshold.png')
    plt.tight_layout()
    plt.show()

    return df_diff


### Plot the number of days the model is above the temp threshold for each year of model data
def plot_model_days_above_threshold(temp_daily_max, threshold=30):
    # Extract all years from the model data
    years = np.unique(temp_daily_max['time.year'].values)

    days_above_list = []
    complete_years = []

    for year in years:
        # Select model data for the year
        model_year = temp_daily_max.sel(time=str(year))
        # Convert to pandas Series
        model_tmax = model_year.to_pandas().squeeze()
        # Skip incomplete years
        if len(model_tmax) < 365:
            continue
        # Count days above threshold
        days_above = (model_tmax > threshold).sum()
        days_above_list.append(days_above)
        complete_years.append(year)

    # Create DataFrame
    df_model = pd.DataFrame({
        'Year': complete_years,
        f'Model days > {threshold}°C': days_above_list
    })

    # Plot bar chart
    plt.figure(figsize=(10,5))
    plt.bar(df_model['Year'], df_model[f'Model days > {threshold}°C'], color='orange')
    plt.xlabel("Year")
    plt.ylabel(f"Model days > {threshold}°C")
    plt.title(f"Number of Days Above {threshold}°C – Model Data")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Figures/model_days_above_threshold.png')
    plt.show()

    return df_model


def plot_multiyear_window_max_climatology(temp_daily_max, obs_df, 
                                          start_year=2019, end_year=2024,
                                          threshold=30):
    """
    For each day-of-year (1..365), compute the maximum temperature across all
    years in the window for both observations and model. Produce a 365-day plot,
    shade days above threshold, and display the difference in number of threshold exceedances.
    """

    # -------------------------
    # OBSERVATIONS
    # -------------------------
    obs_df = obs_df.copy()
    obs_df["DATE"] = pd.to_datetime(obs_df["DATE"])
    mask = (obs_df["DATE"].dt.year >= start_year) & (obs_df["DATE"].dt.year <= end_year)
    obs_win = obs_df[mask].set_index("DATE")
    obs_tmax_c = (obs_win["TMAX"] - 32) * 5/9
    obs_by_doy = obs_tmax_c.groupby(obs_tmax_c.index.dayofyear).max()
    obs_curve = obs_by_doy.reindex(range(1, 366))

    # -------------------------
    # MODEL
    # -------------------------
    model_win = temp_daily_max.sel(time=slice(str(start_year), str(end_year)))
    model_series = model_win.to_pandas().squeeze()
    model_by_doy = model_series.groupby(model_series.index.dayofyear).max()
    model_curve = model_by_doy.reindex(range(1, 366))

    # -------------------------
    # PLOTTING
    # -------------------------
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(obs_curve.index, obs_curve.values, label="Observed Max Across Window", color="blue")
    ax.plot(model_curve.index, model_curve.values, label="Model Max Across Window", color="red")
    
    # Threshold line
    ax.axhline(threshold, linestyle="--", color="k", linewidth=1)

    # Shade days above threshold
    obs_above = obs_curve > threshold
    model_above = model_curve > threshold
    ax.fill_between(obs_curve.index, threshold, obs_curve.values,
                    where=obs_above, color="blue", alpha=0.2, label="Obs > Threshold")
    ax.fill_between(model_curve.index, threshold, model_curve.values,
                    where=model_above, color="red", alpha=0.2, label="Model > Threshold")

    # Month ticks
    months = pd.date_range("2001-01-01", "2001-12-31", freq="MS")  # dummy non-leap year
    month_ticks = months.dayofyear
    month_labels = months.month
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

    ax.set_xlabel("Month")
    ax.set_ylabel("Max Temperature Across Window (°C)")
    ax.set_title(f"Multi-Year Maximum Daily Temperature Climatology ({start_year}–{end_year})")
    ax.grid(True)
    ax.legend()

    # -------------------------
    # DAYS ABOVE THRESHOLD
    # -------------------------
    obs_days = obs_above.sum()
    model_days = model_above.sum()
    diff_days = obs_days - model_days  # obs minus model

    # Add difference to upper left corner
    ax.text(0.02, 0.95, f"Obs - Model days above {threshold}°C: {diff_days}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'Figures/multiyear_window_max_climatology_{start_year}_{end_year}.png')
    #plt.show()

    return obs_curve, model_curve, obs_days, model_days, diff_days



###% Procedure
plot_multiyear_window_max_climatology(temp_daily_max, obs_df,
                                          start_year=2015, end_year=2024,
                                          threshold=30)

