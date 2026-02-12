# XArray API Reference

## Overview

XArray is a powerful Python library for working with labeled multi-dimensional arrays. It's particularly well-suited for remote sensing data with dimensions like time, latitude, and longitude.

## Core Data Structures

### DataArray

A multi-dimensional array with labeled dimensions and coordinates.

**Creation:**

```python
import xarray as xr
import numpy as np

# From numpy array
data = np.random.rand(365, 100, 100)
da = xr.DataArray(
    data,
    dims=['time', 'y', 'x'],
    coords={
        'time': pd.date_range('2023-01-01', periods=365),
        'y': np.arange(100),
        'x': np.arange(100)
    },
    name='temperature'
)
```

**Attributes:**

- `dims` - Dimension names
- `coords` - Coordinate arrays
- `values` - Underlying numpy array
- `attrs` - Metadata dictionary
- `name` - Variable name

### Dataset

A dict-like container of DataArrays sharing dimensions.

**Creation:**

```python
ds = xr.Dataset({
    'temperature': (['time', 'y', 'x'], temp_data),
    'precipitation': (['time', 'y', 'x'], precip_data)
},
coords={
    'time': pd.date_range('2023-01-01', periods=365),
    'y': np.arange(100),
    'x': np.arange(100)
})
```

## Selection and Indexing

### Position-based (isel)

Select by integer position:

```python
# Select first time step
da.isel(time=0)

# Select range
da.isel(time=slice(0, 10))

# Select from multiple dimensions
da.isel(time=0, x=50, y=50)

# Negative indexing
da.isel(time=-1)  # Last time step
```

### Label-based (sel)

Select by coordinate value:

```python
# Select by exact value
da.sel(time='2023-01-15')

# Select range
da.sel(time=slice('2023-01-01', '2023-01-31'))

# Nearest neighbor
da.sel(time='2023-01-15', method='nearest')

# Multiple dimensions
da.sel(time='2023-01-15', x=50, y=50)
```

### Boolean Indexing (where)

Filter based on conditions:

```python
# Keep values where condition is True
da.where(da > 0)

# Replace with specific value
da.where(da > 0, other=0)

# Multiple conditions
da.where((da > 0) & (da < 100))
```

## Computation

### Aggregation

```python
# Along all dimensions
da.mean()
da.sum()
da.std()
da.min()
da.max()
da.median()

# Along specific dimension
da.mean(dim='time')
da.sum(dim=['x', 'y'])

# Keep attributes
da.mean(dim='time', keep_attrs=True)

# Skip NaN values
da.mean(dim='time', skipna=True)
```

### Element-wise Operations

```python
# Arithmetic
result = da + 10
result = da * 2
result = da / 100

# Mathematical functions
np.sqrt(da)
np.exp(da)
np.log(da)

# Trigonometric
np.sin(da)
np.cos(da)

# Between DataArrays
result = da1 + da2  # Automatically aligns coordinates
```

### GroupBy Operations

```python
# Group by time components
monthly = da.groupby('time.month').mean()
seasonal = da.groupby('time.season').mean()
yearly = da.groupby('time.year').mean()

# Custom grouping
bins = [0, 10, 20, 30]
grouped = da.groupby_bins('temperature', bins).mean()

# Multiple operations
stats = da.groupby('time.month').agg(['mean', 'std', 'min', 'max'])
```

### Resampling

```python
# Temporal resampling
monthly = da.resample(time='1M').mean()
weekly = da.resample(time='1W').median()
daily = da.resample(time='1D').sum()

# Upsampling with interpolation
hourly = da.resample(time='1H').interpolate('linear')
```

### Rolling Windows

```python
# Moving average
rolling_mean = da.rolling(time=7, center=True).mean()

# Multiple dimensions
rolling_2d = da.rolling(x=3, y=3, center=True).mean()

# Custom function
rolling_custom = da.rolling(time=7).reduce(np.percentile, q=90)
```

## Interpolation

```python
# Linear interpolation
interp = da.interp(time='2023-01-15T12:00:00')

# Multiple points
new_times = pd.date_range('2023-01-01', '2023-12-31', freq='6H')
interp = da.interp(time=new_times)

# Different methods
interp = da.interp(time=new_times, method='cubic')
interp = da.interp(time=new_times, method='nearest')

# Fill NaN
interp = da.interpolate_na(dim='time', method='linear')
```

## Broadcasting

```python
# Automatic alignment
da1 = xr.DataArray([1, 2, 3], dims='x', coords={'x': [0, 1, 2]})
da2 = xr.DataArray([10, 20], dims='y', coords={'y': [0, 1]})

# Result has dimensions (x, y)
result = da1 + da2
```

## Plotting

### Basic Plots

```python
# Line plot (1D)
da.sel(x=50, y=50).plot()

# 2D plot
da.isel(time=0).plot()

# With options
da.isel(time=0).plot(
    cmap='viridis',
    vmin=0,
    vmax=100,
    cbar_kwargs={'label': 'Temperature (°C)'}
)
```

### Advanced Plots

```python
# Contour plot
da.isel(time=0).plot.contour(levels=10)

# Filled contour
da.isel(time=0).plot.contourf(levels=20, cmap='RdYlBu_r')

# Histogram
da.plot.hist(bins=50)

# Multiple subplots
da.isel(time=[0, 10, 20, 30]).plot(col='time', col_wrap=2)
```

## I/O Operations

### NetCDF

```python
# Write
ds.to_netcdf('data.nc')

# Read
ds = xr.open_dataset('data.nc')

# With chunks (lazy loading)
ds = xr.open_dataset('data.nc', chunks={'time': 10})

# Multiple files
ds = xr.open_mfdataset('data_*.nc', combine='by_coords')
```

### Zarr

```python
# Write
ds.to_zarr('data.zarr', mode='w')

# Read
ds = xr.open_zarr('data.zarr')

# Append
ds.to_zarr('data.zarr', append_dim='time')
```

### GeoTIFF (via rioxarray)

```python
import rioxarray as rxr

# Read
da = rxr.open_rasterio('image.tif')

# Write
da.rio.to_raster('output.tif')

# With CRS
da.rio.write_crs('EPSG:4326', inplace=True)
da.rio.to_raster('output.tif')
```

## Dask Integration

### Chunking

```python
# Chunk on creation
ds = xr.Dataset({
    'temperature': (['time', 'y', 'x'], 
                   dask.array.random.random((365, 1000, 1000), 
                   chunks=(10, 100, 100)))
})

# Rechunk existing data
ds_rechunked = ds.chunk({'time': 30, 'x': 500, 'y': 500})

# Auto chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')
```

### Lazy Evaluation

```python
# Operations are lazy
result = ds.mean(dim='time')  # Not computed yet

# Trigger computation
result_computed = result.compute()

# Persist in memory
result_persisted = result.persist()
```

### Parallel Operations

```python
from dask.distributed import Client

client = Client()

# Operations run in parallel
result = ds.mean(dim='time').compute()
```

## Coordinate Operations

### Adding Coordinates

```python
# Add new coordinate
ds = ds.assign_coords(height=100)

# From existing data
ds = ds.assign_coords(ndvi=(ds.nir - ds.red) / (ds.nir + ds.red))
```

### Swapping Dimensions

```python
# Swap dimension with coordinate
ds_swapped = ds.swap_dims({'time': 'day_of_year'})
```

### Multi-index

```python
# Create multi-index
ds = ds.set_index(location=['lat', 'lon'])

# Unstack
ds_unstacked = ds.unstack('location')
```

## Merging and Concatenating

### Merge

```python
# Merge datasets
merged = xr.merge([ds1, ds2])

# With options
merged = xr.merge([ds1, ds2], compat='override')
```

### Concatenate

```python
# Along existing dimension
concat = xr.concat([ds1, ds2], dim='time')

# Create new dimension
concat = xr.concat([ds1, ds2], dim='model')
```

### Combine

```python
# Combine by coordinates
combined = xr.combine_by_coords([ds1, ds2, ds3])

# Combine nested
combined = xr.combine_nested([[ds1, ds2], [ds3, ds4]], 
                             concat_dim=['x', 'y'])
```

## Apply Functions

### apply_ufunc

```python
def custom_function(x, y):
    return x ** 2 + y

result = xr.apply_ufunc(
    custom_function,
    da1, da2,
    dask='parallelized',
    output_dtypes=[float]
)
```

### map_blocks

```python
def process_block(block):
    # Custom processing
    return block * 2

result = da.map_blocks(process_block)
```

## Performance Tips

### 1. Use Chunking

```python
# Good: Chunked for large data
ds = xr.open_dataset('large.nc', chunks={'time': 10})

# Bad: Load all into memory
ds = xr.open_dataset('large.nc')
```

### 2. Avoid Loops

```python
# Good: Vectorized
result = (ds.nir - ds.red) / (ds.nir + ds.red)

# Bad: Loop over time
for t in ds.time:
    result = (ds.nir.sel(time=t) - ds.red.sel(time=t)) / \
             (ds.nir.sel(time=t) + ds.red.sel(time=t))
```

### 3. Use Appropriate Chunks

```python
# Good: Balanced chunks (10-100 MB per chunk)
ds = ds.chunk({'time': 10, 'x': 512, 'y': 512})

# Bad: Too small
ds = ds.chunk({'time': 1, 'x': 10, 'y': 10})

# Bad: Too large
ds = ds.chunk({'time': 1000, 'x': 10000, 'y': 10000})
```

### 4. Persist Intermediate Results

```python
# Good: Persist frequently used results
intermediate = ds.mean(dim='time').persist()
result1 = intermediate + 10
result2 = intermediate * 2

# Bad: Recompute each time
result1 = ds.mean(dim='time') + 10
result2 = ds.mean(dim='time') * 2
```

## Common Patterns

### Calculate NDVI

```python
ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
ndvi = ndvi.rename('NDVI')
```

### Temporal Statistics

```python
# Mean over time
temporal_mean = ds.mean(dim='time')

# Anomalies
climatology = ds.groupby('time.dayofyear').mean()
anomalies = ds.groupby('time.dayofyear') - climatology
```

### Spatial Subsetting

```python
# Bounding box
subset = ds.sel(
    x=slice(xmin, xmax),
    y=slice(ymin, ymax)
)

# Point extraction
point = ds.sel(x=82.5, y=27.0, method='nearest')
```

## Additional Resources

- [XArray Documentation](https://docs.xarray.dev/)
- [XArray Tutorial](https://tutorial.xarray.dev/)
- [XArray GitHub](https://github.com/pydata/xarray)
- [Pangeo Gallery](https://gallery.pangeo.io/)
- [XArray Cheat Sheet](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html)
