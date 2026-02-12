# Time Series Extraction

Time series analysis is a fundamental technique in remote sensing, allowing us to track changes in vegetation, water bodies, land use, and climate over time.

## Overview

In this section, you'll learn how to:

- Extract time series from satellite imagery
- Handle irregular time intervals
- Resample data to regular intervals
- Calculate temporal statistics
- Detect trends and anomalies

## Basic Time Series Extraction

### Point-based Extraction

Extract values at a specific location over time:

```python
import xarray as xr
import matplotlib.pyplot as plt

# Open time series dataset
ds = xr.open_zarr('ndvi_timeseries.zarr')

# Extract at a point (lon, lat)
point_ts = ds.sel(x=82.5, y=27.0, method='nearest')

# Plot time series
point_ts.NDVI.plot(marker='o')
plt.title('NDVI Time Series at Point')
plt.ylabel('NDVI')
plt.grid(True, alpha=0.3)
plt.show()
```

### Regional Extraction

Extract mean values over a region:

```python
# Define bounding box
lon_min, lon_max = 82.0, 83.0
lat_min, lat_max = 26.5, 27.5

# Extract region
region_ts = ds.sel(
    x=slice(lon_min, lon_max),
    y=slice(lat_min, lat_max)
)

# Calculate spatial mean
mean_ts = region_ts.NDVI.mean(dim=['x', 'y'])

# Plot
mean_ts.plot(marker='o')
plt.title('Mean NDVI over Region')
plt.ylabel('NDVI')
plt.show()
```

## Resampling Time Series

### Temporal Aggregation

Resample to regular intervals:

```python
# Resample to monthly
monthly = ds.NDVI.resample(time='1M').mean()

# Resample to weekly
weekly = ds.NDVI.resample(time='1W').median()

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ds.NDVI.mean(dim=['x', 'y']).plot(ax=axes[0], label='Daily')
monthly.mean(dim=['x', 'y']).plot(ax=axes[1], label='Monthly', marker='o')

axes[0].set_title('Original Daily Data')
axes[1].set_title('Monthly Aggregation')
plt.tight_layout()
plt.show()
```

## Temporal Statistics

### Calculate Statistics

```python
# Temporal mean
temporal_mean = ds.NDVI.mean(dim='time')

# Temporal standard deviation
temporal_std = ds.NDVI.std(dim='time')

# Temporal range
temporal_range = ds.NDVI.max(dim='time') - ds.NDVI.min(dim='time')

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

temporal_mean.plot(ax=axes[0], cmap='RdYlGn')
axes[0].set_title('Mean NDVI')

temporal_std.plot(ax=axes[1], cmap='YlOrRd')
axes[1].set_title('NDVI Std Dev')

temporal_range.plot(ax=axes[2], cmap='viridis')
axes[2].set_title('NDVI Range')

plt.tight_layout()
plt.show()
```

## Trend Analysis

### Linear Trend

```python
from scipy import stats
import numpy as np

def calculate_trend(data):
    """Calculate linear trend."""
    x = np.arange(len(data))
    mask = ~np.isnan(data)
    if mask.sum() < 3:
        return np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], data[mask])
    return slope

# Apply to each pixel
trend = xr.apply_ufunc(
    calculate_trend,
    ds.NDVI,
    input_core_dims=[['time']],
    vectorize=True
)

# Visualize
trend.plot(cmap='RdBu_r', center=0)
plt.title('NDVI Trend (slope per time step)')
plt.show()
```

## Next Steps

→ Continue to [Data Aggregation](aggregation.md)

## Additional Resources

- [XArray Time Series](https://docs.xarray.dev/en/stable/user-guide/time-series.html)
- [Pandas Resampling](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling)
