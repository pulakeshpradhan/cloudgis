# Complete Time Series Workflow

## Overview

This practical example demonstrates a complete end-to-end workflow for time series analysis using cloud-native tools. We'll cover three approaches:

1. **Geemap Tiled Download** → Read with XArray
2. **Direct XEE Approach** → Stream from Earth Engine
3. **Dask + Zarr** → Scalable time series processing

## Scenario

**Objective**: Analyze NDVI time series for a region to detect vegetation changes over 2023.

**Area**: Agricultural region in Uttar Pradesh, India  
**Data**: Sentinel-2 monthly composites  
**Output**: Time series analysis with trend detection

## Approach 1: Geemap Tiled Download + XArray

### Step 1: Download Monthly Composites with Geemap

```python
import ee
import geemap
import os
import calendar
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='spatialgeography')

# Create output directory
output_folder = 'timeseries_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define region of interest
roi = ee.Geometry.Rectangle([82.0, 26.5, 82.5, 27.0])

# Download monthly NDVI composites for 2023
print("Downloading monthly NDVI composites...")

monthly_files = []

for month in range(1, 13):
    # Define date range
    start_date = f'2023-{month:02d}-01'
    if month == 12:
        end_date = '2024-01-01'
    else:
        end_date = f'2023-{month+1:02d}-01'
    
    # Get Sentinel-2 data
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    
    # Cloud masking function
    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                     qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)
    
    # Apply cloud mask and create median composite
    s2_masked = s2.map(mask_clouds)
    median = s2_masked.median()
    
    # Calculate NDVI (processing done server-side in Earth Engine)
    ndvi = median.normalizedDifference(['B8', 'B4']).clip(roi)
    
    # Download using geemap (automatic tiling, no EECU usage)
    month_name = calendar.month_name[month]
    output_file = os.path.join(output_folder, f'ndvi_2023_{month:02d}.tif')
    
    try:
        geemap.download_ee_image(
            ndvi,
            filename=output_file,
            region=roi,
            scale=20,  # 20m resolution
            crs='EPSG:4326',
            num_threads=4  # Parallel download
        )
        monthly_files.append(output_file)
        print(f"✓ Downloaded: {month_name}")
    except Exception as e:
        print(f"✗ Failed {month_name}: {str(e)}")

print(f"\nTotal files downloaded: {len(monthly_files)}")
```

### Step 2: Read Downloaded Files with XArray

```python
# Read all monthly files into XArray Dataset
print("\nReading files with XArray...")

# Read first file to get metadata
first_raster = rxr.open_rasterio(monthly_files[0], masked=True)

# Create list to store all monthly data
monthly_data = []
time_coords = []

for i, file in enumerate(monthly_files):
    # Read raster
    raster = rxr.open_rasterio(file, masked=True)
    
    # Extract data (remove band dimension as we only have NDVI)
    data = raster.squeeze('band', drop=True)
    
    # Add to list
    monthly_data.append(data)
    
    # Create time coordinate (middle of month)
    month = i + 1
    time_coords.append(datetime(2023, month, 15))

# Stack along time dimension
ndvi_ts = xr.concat(monthly_data, dim='time')
ndvi_ts = ndvi_ts.assign_coords(time=time_coords)
ndvi_ts.name = 'NDVI'

print(ndvi_ts)
```

### Step 3: Analyze Time Series with XArray

```python
# Calculate statistics
print("\nCalculating statistics...")

# Temporal statistics
ndvi_mean = ndvi_ts.mean(dim='time')
ndvi_std = ndvi_ts.std(dim='time')
ndvi_max = ndvi_ts.max(dim='time')
ndvi_min = ndvi_ts.min(dim='time')

# Spatial mean time series
ndvi_spatial_mean = ndvi_ts.mean(dim=['x', 'y'])

# Calculate trend using linear regression
from scipy import stats

def calculate_trend(data):
    """Calculate linear trend."""
    x = np.arange(len(data))
    mask = ~np.isnan(data)
    if mask.sum() < 3:  # Need at least 3 points
        return np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], data[mask])
    return slope

# Apply trend calculation
trend = xr.apply_ufunc(
    calculate_trend,
    ndvi_ts,
    input_core_dims=[['time']],
    vectorize=True
)

print(f"Mean NDVI: {ndvi_mean.mean().values:.3f}")
print(f"NDVI Std Dev: {ndvi_std.mean().values:.3f}")
```

### Step 4: Visualize Results

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Mean NDVI
ndvi_mean.plot(ax=axes[0, 0], cmap='RdYlGn', vmin=-1, vmax=1)
axes[0, 0].set_title('Mean NDVI (2023)')
axes[0, 0].set_axis_off()

# 2. Standard Deviation
ndvi_std.plot(ax=axes[0, 1], cmap='YlOrRd')
axes[0, 1].set_title('NDVI Standard Deviation')
axes[0, 1].set_axis_off()

# 3. Trend
trend.plot(ax=axes[0, 2], cmap='RdBu_r', center=0)
axes[0, 2].set_title('NDVI Trend (slope)')
axes[0, 2].set_axis_off()

# 4. Time series plot
ndvi_spatial_mean.plot(ax=axes[1, 0], marker='o')
axes[1, 0].set_title('Spatial Mean NDVI Time Series')
axes[1, 0].set_ylabel('NDVI')
axes[1, 0].grid(True, alpha=0.3)

# 5. Seasonal comparison (Jan vs Jul)
ndvi_ts.isel(time=0).plot(ax=axes[1, 1], cmap='RdYlGn', vmin=-1, vmax=1)
axes[1, 1].set_title('January NDVI')
axes[1, 1].set_axis_off()

ndvi_ts.isel(time=6).plot(ax=axes[1, 2], cmap='RdYlGn', vmin=-1, vmax=1)
axes[1, 2].set_title('July NDVI')
axes[1, 2].set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'ndvi_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Results saved to {output_folder}/")
```

## Approach 2: Direct XEE Approach (Streaming)

### Stream Time Series Directly from Earth Engine

```python
import xarray as xr
import xee
import ee

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define region and time range
roi = ee.Geometry.Rectangle([82.0, 26.5, 82.5, 27.0])

# Get Sentinel-2 ImageCollection
s2_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

# Cloud masking
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Apply cloud mask
s2_masked = s2_collection.map(mask_s2_clouds)

# Open as XArray Dataset using XEE (streaming, no download!)
print("Opening Earth Engine data with XEE...")

ds_xee = xr.open_dataset(
    s2_masked,
    engine='ee',
    geometry=roi,
    scale=20,
    crs='EPSG:4326',
    ee_mask_value=-9999
)

# IMPORTANT: Sort by time immediately for resampling to work
ds_xee = ds_xee.sortby('time')
print(ds_xee)

# Calculate NDVI directly on the streamed data
print("\nCalculating NDVI...")
ndvi_xee = (ds_xee.B8 - ds_xee.B4) / (ds_xee.B8 + ds_xee.B4)

# Resample to monthly
print("Resampling to monthly...")
ndvi_monthly_xee = ndvi_xee.resample(time='1M').median()

# Calculate spatial mean
# XEE dimensions are usually lon/lat or X/Y depending on CRS
spatial_dims = [d for d in list(ds_xee.dims) if d not in ['time', 'band']]
ndvi_ts_xee = ndvi_monthly_xee.mean(dim=spatial_dims)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# XEE approach
ndvi_ts_xee.plot(ax=axes[0], marker='o', label='XEE (Streaming)')
axes[0].set_title('NDVI Time Series - XEE Approach')
axes[0].set_ylabel('NDVI')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Downloaded approach (from previous)
ndvi_spatial_mean.plot(ax=axes[1], marker='s', label='Geemap (Downloaded)')
axes[1].set_title('NDVI Time Series - Geemap Approach')
axes[1].set_ylabel('NDVI')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n✓ XEE streaming approach complete!")
```

## Approach 3: Dask + Zarr for Scalable Processing

### Step 1: Load Data with Dask

```python
from dask.distributed import Client, LocalCluster
import dask.array as da

# Start Dask cluster
print("Starting Dask cluster...")
cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB')
client = Client(cluster)
print(client)

# Load downloaded files with Dask chunking
print("\nLoading data with Dask...")

# Read files with chunking
monthly_data_dask = []
time_coords = []

for i, file in enumerate(monthly_files):
    # Read with rioxarray and chunk
    raster = rxr.open_rasterio(file, masked=True, chunks={'x': 256, 'y': 256})
    data = raster.squeeze('band', drop=True)
    monthly_data_dask.append(data)
    
    month = i + 1
    time_coords.append(datetime(2023, month, 15))

# Stack with Dask
ndvi_dask = xr.concat(monthly_data_dask, dim='time')
ndvi_dask = ndvi_dask.assign_coords(time=time_coords)
ndvi_dask = ndvi_dask.chunk({'time': 3, 'x': 256, 'y': 256})
ndvi_dask.name = 'NDVI'

print(ndvi_dask)
print(f"\nDask chunks: {ndvi_dask.chunks}")
```

### Step 2: Save to Zarr (Cloud-Optimized)

```python
import zarr
from numcodecs import Blosc

# Configure compression
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# Save to Zarr
zarr_path = os.path.join(output_folder, 'ndvi_timeseries.zarr')

print(f"\nSaving to Zarr: {zarr_path}")

# Convert to dataset and save
ndvi_dataset = ndvi_dask.to_dataset()

# Add metadata
ndvi_dataset.attrs['title'] = 'NDVI Time Series 2023'
ndvi_dataset.attrs['source'] = 'Sentinel-2 SR'
ndvi_dataset.attrs['region'] = 'Uttar Pradesh, India'
ndvi_dataset.attrs['resolution'] = '20m'

# Save with compression
encoding = {
    'NDVI': {
        'compressor': compressor,
        'chunks': (3, 256, 256)
    }
}

ndvi_dataset.to_zarr(
    zarr_path,
    mode='w',
    encoding=encoding,
    consolidated=True
)

print("✓ Saved to Zarr!")

# Check file size
import shutil
zarr_size = sum(f.stat().st_size for f in Path(zarr_path).rglob('*') if f.is_file())
print(f"Zarr archive size: {zarr_size / 1e6:.2f} MB")
```

### Step 3: Load from Zarr and Process with Dask

```python
# Load from Zarr (instant, lazy loading)
print("\nLoading from Zarr...")
ndvi_from_zarr = xr.open_zarr(zarr_path, consolidated=True)

print(ndvi_from_zarr)

# Perform computations with Dask
print("\nPerforming Dask computations...")

# 1. Calculate anomalies
climatology = ndvi_from_zarr.NDVI.mean(dim='time')
anomalies = ndvi_from_zarr.NDVI - climatology

# 2. Calculate rolling mean (smoothing)
rolling_mean = ndvi_from_zarr.NDVI.rolling(time=3, center=True).mean()

# 3. Calculate percentiles
percentile_10 = ndvi_from_zarr.NDVI.quantile(0.1, dim='time')
percentile_90 = ndvi_from_zarr.NDVI.quantile(0.9, dim='time')

# Compute all at once (parallel with Dask)
print("Computing results...")
results = xr.Dataset({
    'anomalies': anomalies,
    'rolling_mean': rolling_mean,
    'p10': percentile_10,
    'p90': percentile_90
})

# Trigger computation
results_computed = results.compute()

print("✓ Computations complete!")
```

### Step 4: Advanced Time Series Analysis

```python
# Seasonal decomposition
print("\nPerforming seasonal analysis...")

# Group by season
seasonal_mean = ndvi_from_zarr.NDVI.groupby('time.season').mean()

# Monthly statistics
monthly_stats = ndvi_from_zarr.NDVI.groupby('time.month').agg(['mean', 'std', 'min', 'max'])

# Compute
seasonal_computed = seasonal_mean.compute()
monthly_stats_computed = monthly_stats.compute()

# Visualize seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot each season
seasons = ['DJF', 'MAM', 'JJA', 'SON']
for idx, season in enumerate(seasons):
    ax = axes[idx // 2, idx % 2]
    if season in seasonal_computed.season:
        seasonal_computed.sel(season=season).plot(
            ax=ax, cmap='RdYlGn', vmin=-1, vmax=1
        )
        ax.set_title(f'{season} Mean NDVI')
        ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'seasonal_ndvi.png'), dpi=300)
plt.show()
```

### Step 5: Export Results

```python
# Save processed results to Zarr
results_path = os.path.join(output_folder, 'ndvi_analysis_results.zarr')

print(f"\nSaving analysis results to: {results_path}")

results_computed.to_zarr(
    results_path,
    mode='w',
    consolidated=True
)

# Also save as NetCDF for compatibility
netcdf_path = os.path.join(output_folder, 'ndvi_timeseries.nc')
ndvi_from_zarr.to_netcdf(netcdf_path)

print(f"✓ Saved to NetCDF: {netcdf_path}")

# Close Dask client
client.close()
cluster.close()

print("\n✓ All processing complete!")
```

## Complete Workflow Comparison

### Summary Table

| Approach | Pros | Cons | Best For |
| --- | --- | --- | --- |
| **Geemap Download** | Full local control, offline analysis | Requires storage, download time | Repeated analysis, offline work |
| **XEE Streaming** | No storage needed, always latest data | Requires internet, slower for repeated queries | Exploratory analysis, prototyping |
| **Dask + Zarr** | Scalable, efficient, cloud-ready | Initial setup complexity | Large-scale analysis, production |

### Performance Comparison

```python
import time

# Benchmark each approach
print("\n=== Performance Comparison ===\n")

# 1. Geemap + XArray
start = time.time()
result1 = ndvi_ts.mean(dim='time').values
time1 = time.time() - start
print(f"Geemap + XArray: {time1:.2f} seconds")

# 2. XEE (if loaded)
start = time.time()
result2 = ndvi_monthly_xee.mean(dim='time').compute().values
time2 = time.time() - start
print(f"XEE Streaming: {time2:.2f} seconds")

# 3. Dask + Zarr
start = time.time()
result3 = ndvi_from_zarr.NDVI.mean(dim='time').compute().values
time3 = time.time() - start
print(f"Dask + Zarr: {time3:.2f} seconds")

print(f"\nFastest approach: ", end="")
times = {'Geemap': time1, 'XEE': time2, 'Zarr': time3}
print(min(times, key=times.get))
```

## Complete Example Script

Here's the full script combining all approaches:

```python
#!/usr/bin/env python3
"""
Complete Time Series Analysis Workflow
Demonstrates: Geemap Download → XArray → XEE → Dask → Zarr
"""

import ee
import geemap
import xarray as xr
import xee
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
from pathlib import Path
import os
import calendar
from datetime import datetime

# Configuration
ROI = ee.Geometry.Rectangle([82.0, 26.5, 82.5, 27.0])
YEAR = 2023
OUTPUT_FOLDER = 'timeseries_analysis'
SCALE = 20  # meters

def main():
    """Run complete workflow."""
    
    # Initialize
    ee.Initialize(project='spatialgeography')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("="*60)
    print("CLOUD-NATIVE TIME SERIES ANALYSIS WORKFLOW")
    print("="*60)
    
    # Approach 1: Geemap Download
    print("\n[1/3] Geemap Tiled Download...")
    monthly_files = download_monthly_ndvi(ROI, YEAR, OUTPUT_FOLDER)
    ndvi_xarray = load_with_xarray(monthly_files)
    analyze_timeseries(ndvi_xarray, OUTPUT_FOLDER)
    
    # Approach 2: XEE Streaming
    print("\n[2/3] XEE Streaming...")
    ndvi_xee = stream_with_xee(ROI, YEAR)
    compare_approaches(ndvi_xarray, ndvi_xee, OUTPUT_FOLDER)
    
    # Approach 3: Dask + Zarr
    print("\n[3/3] Dask + Zarr Processing...")
    process_with_dask_zarr(monthly_files, OUTPUT_FOLDER)
    
    print("\n" + "="*60)
    print("✓ WORKFLOW COMPLETE!")
    print(f"Results saved to: {OUTPUT_FOLDER}/")
    print("="*60)

def download_monthly_ndvi(roi, year, output_folder):
    """Download monthly NDVI using geemap."""
    # Implementation from Approach 1
    pass

def load_with_xarray(files):
    """Load files with XArray."""
    # Implementation from Approach 1
    pass

def analyze_timeseries(data, output_folder):
    """Analyze time series."""
    # Implementation from Approach 1
    pass

def stream_with_xee(roi, year):
    """Stream data with XEE."""
    # Implementation from Approach 2
    pass

def compare_approaches(data1, data2, output_folder):
    """Compare different approaches."""
    # Implementation from comparison section
    pass

def process_with_dask_zarr(files, output_folder):
    """Process with Dask and save to Zarr."""
    # Implementation from Approach 3
    pass

if __name__ == '__main__':
    main()
```

## Key Takeaways

!!! success "What You Learned"
    - **Geemap**: Download large time series without EECU, automatic tiling
    - **XArray**: Powerful time series analysis with labeled dimensions
    - **XEE**: Stream Earth Engine data directly without downloads
    - **Dask**: Parallel processing for large datasets
    - **Zarr**: Cloud-optimized storage with compression
    - **Integration**: Combine tools for optimal workflow
    - **Performance**: Choose the right tool for your use case

## Next Steps

- Next: [Time Series and Phenological Methods](timeseries-phenology.md)
- Learn [Optimization Techniques](../advanced/optimization.md) for better performance
- Back: [Classification Methods](classification-methods.md)

## Additional Resources

- [geemap Examples](https://geemap.org/notebooks/)
- [XArray Time Series](https://docs.xarray.dev/en/stable/user-guide/time-series.html)
- [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)
- [Zarr Tutorial](https://zarr.readthedocs.io/en/stable/tutorial.html)
