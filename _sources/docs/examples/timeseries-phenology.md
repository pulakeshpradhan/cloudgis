# Time Series and Phenological Methods

Analyze temporal patterns, extract growth cycles, and simulate crop dynamics using XEE, XArray, and Scipy.

## Overview

This example covers:

1. **Time Series Analysis**: STL Decomposition and Smoothing.
2. **Similarity & Classification**: Comparing pixel trajectories.
3. **Phenological Extraction**: Determining SOS (Start of Season) and EOS (End of Season).
4. **Crop Growth Models**: Introduction to yield simulation concepts.

## Step 1: Load Dense Time Series (MODIS)

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Agricultural region
roi = ee.Geometry.Point([76.5, 30.5]).buffer(5000).bounds()

# Load MODIS NDVI
modis = ee.ImageCollection("MODIS/061/MOD13A1") \
    .filterBounds(roi) \
    .filterDate('2020-01-01', '2023-12-31') \
    .select('NDVI')

ds = xr.open_dataset(modis, engine='ee', geometry=roi, scale=500).compute()
ds['NDVI'] = ds.NDVI * 0.0001 # Correcting scale factor
```

## Step 2: Time Series Smoothing (Savitzky-Golay)

Remote sensing time series often have noise. Smoothing is the first step for phenology.

```python
def smooth_time_series(da):
    return xr.apply_ufunc(
        savgol_filter, da,
        kwargs={'window_length': 7, 'polyorder': 2, 'axis': 0},
        dask='parallelized'
    )

ds['NDVI_smooth'] = smooth_time_series(ds.NDVI)

# Plot a single pixel profile
ds.NDVI.isel(lat=5, lon=5).plot(label='Raw', alpha=0.5)
ds.NDVI_smooth.isel(lat=5, lon=5).plot(label='Smoothed', linewidth=2)
plt.legend()
plt.title("NDVI Smoothing for Phenology")
plt.show()
```

## Step 3: Phenological Extraction (Threshold Method)

Determining the Start of Season (SOS) and End of Season (EOS).

```python
def extract_phenology(profile, threshold=0.4):
    """Simple threshold-based SOS extraction."""
    # Find indices where NDVI crosses threshold
    start_season = np.where(profile > threshold)[0]
    if len(start_season) > 0:
        return start_season[0] # Index of SOS
    return np.nan

# Apply to the spatial dataset
sos_map = xr.apply_ufunc(
    extract_phenology, ds.NDVI_smooth,
    input_core_dims=[['time']],
    vectorize=True
)

plt.figure(figsize=(10, 8))
sos_map.plot(cmap='viridis')
plt.title("Start of Season (SOS) Day of Year")
plt.show()
```

## Step 4: Time Series Similarity (Dynamic Time Warping)

Identifying pixels with similar growth patterns is often better achieved via [Dynamic Time Warping](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html) (DTW) rather than Euclidean distance, as it accounts for temporal shifts.

```python
from tslearn.metrics import dtw

# Compare two pixel profiles
p1 = ds.NDVI_smooth.isel(lat=5, lon=5).values.reshape(-1, 1)
p2 = ds.NDVI_smooth.isel(lat=10, lon=10).values.reshape(-1, 1)

dtw_score = dtw(p1, p2)
print(f"Dynamic Time Warping Distance: {dtw_score:.3f}")
```

## Step 5: Applications in Agriculture

Integrating phenology with growing degree days (GDD) for yield simulation.

```python
# Conceptual: Yield = f(Cumulative NDVI, Area, GDD)
yield_proxy = ds.NDVI_smooth.integrate('time')

plt.figure(figsize=(10, 8))
yield_proxy.plot(cmap='YlGn')
plt.title("Cumulative Productivity (Yield Proxy)")
plt.show()
```

## Key Takeaways

!!! success "Summary"
    - **Smoothing**: Essential for removing cloud artifacts before phenology extraction.
    - **Methods**: Threshold-based methods are simple; derivative-based methods are more precise.
    - **Analysis**: XArray makes it easy to integrate temporal profiles across large areas.

→ Next: [Network and Flow Analysis](network-flow.md)
