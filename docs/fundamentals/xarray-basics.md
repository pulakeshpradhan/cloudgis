# XArray Basics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spatialthoughts/courses/blob/master/code/python_remote_sensing/01_xarray_basics.ipynb)

## Overview

[XArray](https://docs.xarray.dev/en/stable/) has emerged as one of the key Python libraries to work with gridded raster datasets. It can natively handle time-series data, making it ideal for working with Remote Sensing datasets.

**Key Features:**

- Builds on NumPy/Pandas for fast arrays and indexing
- Orders of magnitude faster than other Python libraries like `rasterio`
- Growing ecosystem: `rioxarray`, `xarray-spatial`, `XEE`
- Seamlessly works with local and cloud-hosted datasets
- Supports various optimized data formats (NetCDF, Zarr, COG)

In this section, we'll learn XArray basics and create a median composite image from Sentinel-2 time-series data.

## Setup and Data Download

Install required packages:

```python
%%capture
if 'google.colab' in str(get_ipython()):
    !pip install pystac-client odc-stac rioxarray dask botocore
```

Import libraries:

```python
import os
import matplotlib.pyplot as plt
import pystac_client
from odc.stac import stac_load, configure_s3_access
import xarray as xr
import rioxarray as rxr
```

Create working directories:

```python
data_folder = 'data'
output_folder = 'output'

if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
```

## Get Satellite Imagery

Define location and time of interest:

```python
latitude = 27.163
longitude = 82.608
year = 2023
```

Search for Sentinel-2 imagery using STAC:

```python
catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1')

# Configure S3 access for unsigned requests
configure_s3_access(
    aws_unsigned=True,
)

# Define bounding box
km2deg = 1.0 / 111
x, y = (longitude, latitude)
r = 1 * km2deg  # radius in degrees
bbox = (x - r, y - r, x + r, y + r)

# Search catalog
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime=f'{year}',
    query={'eo:cloud_cover': {'lt': 30}},
)
items = search.item_collection()
```

Load images as XArray Dataset:

```python
ds = stac_load(
    items,
    bands=['red', 'green', 'blue', 'nir'],
    resolution=10,
    bbox=bbox,
    chunks={},  # <-- use Dask
    groupby='solar_day',
)
ds
```

Compute the dataset (load into memory):

```python
%%time
ds = ds.compute()
```

## XArray Terminology

A Dataset consists of several components:

### Variables

Similar to bands in a raster dataset. Each variable contains an array of values.

### Dimensions

Similar to array axes (e.g., time, x, y).

### Coordinates

Labels for values in each dimension (e.g., timestamps, latitude, longitude).

### Attributes

Metadata associated with the dataset.

![XArray Terminology](https://courses.spatialthoughts.com/images/common/xarray_terminology.png)

### DataArray

A Dataset consists of one or more `xarray.DataArray` objects. Access variables using dot notation:

```python
da = ds.red
da
```

## Selecting Data

XArray provides powerful selection methods similar to Pandas.

### Index-based Selection (isel)

Select by position using `isel()`:

```python
# Select last time step
da.isel(time=-1)
```

Get values as NumPy array:

```python
da.isel(time=-1).values
```

Select across multiple dimensions:

```python
da.isel(time=-1, x=-1, y=-1).values
```

### Label-based Selection (sel)

View coordinate values:

```python
dates = da.time.values
dates
```

Select by coordinate value:

```python
da.sel(time='2023-12-16')
```

### Nearest Neighbor Lookup

Find closest match when exact value doesn't exist:

```python
da.sel(time='2023-01-01', method='nearest')
```

!!! tip "Interpolation"
    Use `interp()` instead of `sel()` to interpolate values:
    ```python
    da.interp(time='2023-01-01')
    ```

### Range Selection

Select time ranges using `slice()`:

```python
# Select all observations in January 2023
da.sel(time=slice('2023-01-01', '2023-01-31'))
```

## Aggregating Data

XArray makes it easy to aggregate data across dimensions.

### Temporal Aggregation

Create a median composite from all images:

```python
median = ds.median(dim='time')
median
```

### Other Aggregation Functions

```python
# Mean
mean = ds.mean(dim='time')

# Maximum
maximum = ds.max(dim='time')

# Standard deviation
std = ds.std(dim='time')

# Sum
total = ds.sum(dim='time')
```

### GroupBy Operations

Group by time periods:

```python
# Monthly median
monthly = ds.groupby('time.month').median(dim='time')

# Yearly mean
yearly = ds.groupby('time.year').mean(dim='time')

# Seasonal aggregation
seasonal = ds.groupby('time.season').mean(dim='time')
```

## Visualizing Data

Convert Dataset to DataArray for plotting:

```python
median_da = median.to_array('band')
median_da
```

### Basic Plotting

Use `robust=True` for automatic contrast stretching (2nd and 98th percentiles):

```python
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 5)
median_da.sel(band=['red', 'green', 'blue']).plot.imshow(
    ax=ax,
    robust=True)
ax.set_title('RGB Visualization')
ax.set_axis_off()
ax.set_aspect('equal')
plt.show()
```

### Custom Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Red band
median.red.plot(ax=axes[0, 0], cmap='Reds', robust=True)
axes[0, 0].set_title('Red Band')

# Green band
median.green.plot(ax=axes[0, 1], cmap='Greens', robust=True)
axes[0, 1].set_title('Green Band')

# Blue band
median.blue.plot(ax=axes[1, 0], cmap='Blues', robust=True)
axes[1, 0].set_title('Blue Band')

# NIR band
median.nir.plot(ax=axes[1, 1], cmap='YlOrRd', robust=True)
axes[1, 1].set_title('NIR Band')

plt.tight_layout()
plt.show()
```

### RGB Composite

```python
# Create RGB composite
rgb = median_da.sel(band=['red', 'green', 'blue'])

# Normalize to 0-1 range
rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
rgb_norm.plot.imshow(ax=ax)
ax.set_title('True Color Composite')
ax.set_axis_off()
plt.show()
```

## Mathematical Operations

XArray supports element-wise operations:

```python
# Calculate NDVI
ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ndvi.isel(time=0).plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_title('NDVI')
plt.show()
```

### Broadcasting

Operations automatically broadcast across dimensions:

```python
# Subtract mean from each time step
anomaly = ds - ds.mean(dim='time')

# Normalize by standard deviation
normalized = (ds - ds.mean(dim='time')) / ds.std(dim='time')
```

## Saving Data

### NetCDF Format

```python
# Save to NetCDF
output_file = os.path.join(output_folder, 'median_composite.nc')
median.to_netcdf(output_file)

# Load back
loaded = xr.open_dataset(output_file)
```

### GeoTIFF Format

```python
# Save single band as GeoTIFF
output_file = os.path.join(output_folder, 'red_band.tif')
median.red.rio.to_raster(output_file)

# Save RGB composite
rgb_file = os.path.join(output_folder, 'rgb_composite.tif')
median_da.sel(band=['red', 'green', 'blue']).rio.to_raster(rgb_file)
```

### Zarr Format

```python
# Save to Zarr (cloud-optimized)
zarr_file = os.path.join(output_folder, 'median.zarr')
median.to_zarr(zarr_file, mode='w')

# Load back
loaded_zarr = xr.open_zarr(zarr_file)
```

## Advanced Features

### Lazy Loading

XArray supports lazy loading for large datasets:

```python
# Data is not loaded into memory
ds_lazy = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Computation is lazy
result = ds_lazy.mean(dim='time')

# Trigger computation
result_computed = result.compute()
```

### Parallel Computing with Dask

```python
from dask.distributed import Client

# Start Dask client
client = Client()

# Load data with chunks
ds_dask = xr.open_dataset('large_file.nc', chunks={'time': 10, 'x': 512, 'y': 512})

# Parallel computation
result = ds_dask.mean(dim='time').compute()
```

### Resampling

```python
# Resample to monthly data
monthly = ds.resample(time='1M').mean()

# Resample to weekly data
weekly = ds.resample(time='1W').median()
```

### Rolling Windows

```python
# 3-month rolling mean
rolling_mean = ds.rolling(time=3, center=True).mean()

# 5-day rolling median
rolling_median = ds.rolling(time=5, center=True).median()
```

## Exercise

Display the median composite for the month of May.

The snippet below aggregates the time-series to monthly median composites using `groupby()`:

```python
monthly = ds.groupby('time.month').median(dim='time')
monthly
```

You now have a new dimension named `month`. Start your exercise by:

1. Converting the Dataset to a DataArray
2. Extracting data for May (month=5) using `sel()`
3. Plotting the RGB composite

**Solution:**

```python
# Convert to DataArray
monthly_da = monthly.to_array('band')

# Select May
may = monthly_da.sel(month=5)

# Plot RGB
fig, ax = plt.subplots(figsize=(8, 8))
may.sel(band=['red', 'green', 'blue']).plot.imshow(ax=ax, robust=True)
ax.set_title('May Median Composite')
ax.set_axis_off()
ax.set_aspect('equal')
plt.show()
```

## Key Takeaways

!!! success "What You Learned"
    - XArray provides labeled, multi-dimensional arrays perfect for satellite imagery
    - Use `isel()` for index-based selection and `sel()` for label-based selection
    - Aggregation functions like `mean()`, `median()`, `max()` work across dimensions
    - `groupby()` enables temporal aggregations (monthly, yearly, seasonal)
    - Visualization is straightforward with built-in plotting methods
    - XArray integrates seamlessly with Dask for parallel computing
    - Multiple output formats supported: NetCDF, GeoTIFF, Zarr

## Next Steps

→ Continue to [STAC and Dask Basics](stac-dask.md)

## Additional Resources

- [XArray Documentation](https://docs.xarray.dev/)
- [XArray Tutorial](https://tutorial.xarray.dev/)
- [Rioxarray Documentation](https://corteva.github.io/rioxarray/)
- [XArray Plotting](https://docs.xarray.dev/en/stable/user-guide/plotting.html)
- [Pangeo Gallery](https://gallery.pangeo.io/)
