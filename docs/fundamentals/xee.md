# XEE for Earth Engine

## Overview

[XEE (XArray Earth Engine Extension)](https://github.com/google/Xee) enables you to use Google Earth Engine datasets with XArray. This integration brings together the massive Earth Engine data catalog with the powerful analysis capabilities of XArray and Dask.

**Key Features:**

- Access Earth Engine ImageCollections as XArray datasets
- Leverage Earth Engine's petabyte-scale data catalog
- Use XArray's intuitive API for analysis
- Combine with Dask for parallel processing
- Seamless integration with existing XArray workflows

## Prerequisites

### Earth Engine Account

1. Sign up at [earthengine.google.com](https://earthengine.google.com/)
2. Wait for approval (usually instant for research/education)

### Authentication

```python
import ee

# Authenticate (first time only)
ee.Authenticate()

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')
```

For Colab:

```python
# Authenticate in Colab
ee.Authenticate()
ee.Initialize(project='spatialgeography')
```

## Installation

```python
%%capture
!pip install xee earthengine-api
```

## Basic Usage

### Opening an ImageCollection

```python
import xarray as xr
import ee

ee.Initialize(project='spatialgeography')

# Open Sentinel-2 as XArray dataset
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Point([lon, lat]).buffer(10000),
    scale=10
)

ds
```

### Specifying Time Range

```python
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Point([82.6, 27.2]).buffer(10000),
    scale=10,
    ee_mask_value=-9999,
    # Time range
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

### Selecting Bands

```python
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Point([82.6, 27.2]).buffer(10000),
    scale=10,
    # Select specific bands
    variables=['B4', 'B8', 'B11']  # Red, NIR, SWIR
)
```

## Working with Different Datasets

### Landsat

```python
# Landsat 8
ds_l8 = xr.open_dataset(
    'ee://LANDSAT/LC08/C02/T1_L2',
    engine='ee',
    geometry=ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]),
    scale=30,
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

### MODIS

```python
# MODIS NDVI
ds_modis = xr.open_dataset(
    'ee://MODIS/006/MOD13A2',
    engine='ee',
    geometry=ee.Geometry.Point([lon, lat]).buffer(50000),
    scale=1000,
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

### Climate Data

```python
# ERA5 Climate Reanalysis
ds_era5 = xr.open_dataset(
    'ee://ECMWF/ERA5/DAILY',
    engine='ee',
    geometry=ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]),
    scale=27830,  # ~25km
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

### Terrain Data

```python
# SRTM Digital Elevation Model
dem = xr.open_dataset(
    'ee://USGS/SRTMGL1_003',
    engine='ee',
    geometry=ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]),
    scale=30
)
```

## Advanced Features

### Cloud Masking

```python
def mask_clouds(image):
    """Mask clouds in Sentinel-2 imagery."""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Apply cloud mask
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(ee.Geometry.Point([lon, lat])) \
    .filterDate('2023-01-01', '2023-12-31') \
    .map(mask_clouds)

# Open as XArray
ds = xr.open_dataset(
    collection,
    engine='ee',
    geometry=ee.Geometry.Point([lon, lat]).buffer(10000),
    scale=10
)
```

### Filtering Collections

```python
# Filter by cloud cover
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(ee.Geometry.Point([lon, lat])) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

ds = xr.open_dataset(
    collection,
    engine='ee',
    geometry=ee.Geometry.Point([lon, lat]).buffer(10000),
    scale=10
)
```

### Custom Geometries

```python
# Polygon geometry
polygon = ee.Geometry.Polygon([[
    [lon_min, lat_min],
    [lon_max, lat_min],
    [lon_max, lat_max],
    [lon_min, lat_max],
    [lon_min, lat_min]
]])

ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=polygon,
    scale=10,
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

### Using Feature Collections

```python
# Load administrative boundaries
countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
india = countries.filter(ee.Filter.eq('country_na', 'India'))

# Get geometry
geometry = india.geometry()

# Open dataset for India
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=geometry,
    scale=100,  # Use coarser resolution for large areas
    start_time='2023-01-01',
    end_time='2023-01-31'
)
```

## Combining with XArray Operations

### Calculate Spectral Indices

```python
# Load Sentinel-2 data
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Point([82.6, 27.2]).buffer(10000),
    scale=10,
    start_time='2023-01-01',
    end_time='2023-12-31',
    variables=['B4', 'B8']  # Red, NIR
)

# Calculate NDVI
ndvi = (ds.B8 - ds.B4) / (ds.B8 + ds.B4)

# Visualize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ndvi.isel(time=0).plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_title('NDVI from Earth Engine')
plt.show()
```

### Time Series Analysis

```python
# Load MODIS NDVI
ds = xr.open_dataset(
    'ee://MODIS/006/MOD13A2',
    engine='ee',
    geometry=ee.Geometry.Point([82.6, 27.2]).buffer(5000),
    scale=1000,
    start_time='2020-01-01',
    end_time='2023-12-31'
)

# Extract NDVI time series
# IMPORTANT: Always sort by time before resampling or extracting time series
ndvi_ts = ds.NDVI.sortby('time').mean(dim=['X', 'Y'])

# Plot time series
fig, ax = plt.subplots(figsize=(12, 6))
ndvi_ts.plot(ax=ax)
ax.set_title('NDVI Time Series')
ax.set_ylabel('NDVI')
plt.show()
```

### Important Tips for XEE

!!! tip "Crucial: Always Sort by Time"
    Earth Engine collections are not always returned in chronological order. To avoid `ValueError: Index must be monotonic for resampling`, always sort your dataset:
    ```python
    ds = ds.sortby('time')
    ```

!!! tip "Spatial Alignment & Projections"
    For consistent results and global compatibility, always specify the coordinate system as **EPSG:4326** (WGS 84).
    ```python
    ds = xr.open_dataset(..., crs='EPSG:4326', scale=100)
    ```

!!! tip "Handling Sparse Data (NaNs)"
    Aggressive cloud masking can remove all data for certain periods. Always check if your dataset contains valid numeric data before plotting:
    ```python
    if ds.NDVI.notnull().any():
        ds.NDVI.plot()
    else:
        print("No valid data found in this period.")
    ```

### Spatial Aggregation

```python
# Load temperature data
ds = xr.open_dataset(
    'ee://ECMWF/ERA5/DAILY',
    engine='ee',
    geometry=ee.Geometry.Rectangle([70, 8, 97, 35]),  # India bounds
    scale=27830,
    start_time='2023-01-01',
    end_time='2023-12-31',
    variables=['mean_2m_air_temperature']
)

# Calculate spatial mean
temp_mean = ds.mean_2m_air_temperature.mean(dim=['lon', 'lat'])

# Convert from Kelvin to Celsius
temp_celsius = temp_mean - 273.15

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
temp_celsius.plot(ax=ax)
ax.set_title('Mean Temperature over India')
ax.set_ylabel('Temperature (°C)')
plt.show()
```

## Integration with Dask

```python
from dask.distributed import Client

# Start Dask client
client = Client()

# Open large dataset with chunking
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]),
    scale=10,
    start_time='2023-01-01',
    end_time='2023-12-31',
    chunks={'time': 10, 'X': 512, 'Y': 512}
)

# Parallel computation
result = ds.mean(dim='time').compute()
```

## Real-World Example: Land Cover Change

```python
import ee
import xarray as xr
import matplotlib.pyplot as plt

ee.Initialize(project='spatialgeography')

# Define area of interest
aoi = ee.Geometry.Point([82.6, 27.2]).buffer(20000)

# Load Sentinel-2 for two time periods
ds_2020 = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=aoi,
    scale=20,
    start_time='2020-01-01',
    end_time='2020-12-31',
    variables=['B4', 'B8']
)

ds_2023 = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=aoi,
    scale=20,
    start_time='2023-01-01',
    end_time='2023-12-31',
    variables=['B4', 'B8']
)

# Calculate median NDVI for each period
ndvi_2020 = ((ds_2020.B8 - ds_2020.B4) / (ds_2020.B8 + ds_2020.B4)).median(dim='time')
ndvi_2023 = ((ds_2023.B8 - ds_2023.B4) / (ds_2023.B8 + ds_2023.B4)).median(dim='time')

# Calculate change
ndvi_change = ndvi_2023 - ndvi_2020

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ndvi_2020.plot(ax=axes[0], cmap='RdYlGn', vmin=-1, vmax=1)
axes[0].set_title('NDVI 2020')

ndvi_2023.plot(ax=axes[1], cmap='RdYlGn', vmin=-1, vmax=1)
axes[1].set_title('NDVI 2023')

ndvi_change.plot(ax=axes[2], cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[2].set_title('NDVI Change (2023-2020)')

plt.tight_layout()
plt.show()
```

## Performance Tips

### 1. Use Appropriate Scale

```python
# Too fine - slow and large
ds = xr.open_dataset(..., scale=10)  # 10m

# Appropriate for analysis
ds = xr.open_dataset(..., scale=30)  # 30m

# Coarse for overview
ds = xr.open_dataset(..., scale=100)  # 100m
```

### 2. Limit Spatial Extent

```python
# Use smallest necessary geometry
geometry = ee.Geometry.Point([lon, lat]).buffer(5000)  # 5km buffer
```

### 3. Filter Before Loading

```python
# Filter in Earth Engine (server-side)
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(geometry) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

ds = xr.open_dataset(collection, engine='ee', ...)
```

### 4. Select Only Needed Bands

```python
# Don't load all bands
ds = xr.open_dataset(..., variables=['B4', 'B8'])  # Only Red and NIR
```

## Troubleshooting

### Authentication Issues

```python
# Re-authenticate
ee.Authenticate(force=True)
ee.Initialize(project='spatialgeography')
```

### Memory Errors

```python
# Use coarser resolution
ds = xr.open_dataset(..., scale=100)  # Instead of 10

# Or smaller area
geometry = ee.Geometry.Point([lon, lat]).buffer(1000)  # 1km instead of 10km
```

### Slow Performance

```python
# Use Dask chunking
ds = xr.open_dataset(..., chunks={'time': 5, 'X': 256, 'Y': 256})

# Reduce spatial extent
# Increase scale (lower resolution)
```

## Key Takeaways

!!! success "What You Learned"
    - XEE enables XArray access to Earth Engine datasets
    - Combine Earth Engine's data catalog with XArray's analysis tools
    - Use Earth Engine for server-side filtering and processing
    - Integrate with Dask for parallel computation
    - Access diverse datasets: Sentinel, Landsat, MODIS, climate data
    - Apply cloud masking and custom filters
    - Perform time series and spatial analysis

## Next Steps

→ Continue to [Calculating Spectral Indices](../processing/spectral-indices.md)

## Additional Resources

- [XEE Documentation](https://github.com/google/Xee)
- [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)
- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [Earth Engine Guides](https://developers.google.com/earth-engine/guides)
- [XArray Earth Engine Examples](https://github.com/google/Xee/tree/main/examples)
