# Data Aggregation

Aggregating remote sensing data across spatial and temporal dimensions is essential for regional analysis, climate studies, and change detection.

## Overview

Learn how to:

- Aggregate data spatially (zonal statistics)
- Aggregate data temporally (composites)
- Combine spatial and temporal aggregation
- Use groupby operations for categorical analysis

## Spatial Aggregation

### Zonal Statistics

Calculate statistics within zones:

```python
import xarray as xr
import geopandas as gpd

# Load data
ds = xr.open_zarr('ndvi_data.zarr')

# Load zones (e.g., administrative boundaries)
zones = gpd.read_file('districts.geojson')

# For each zone, calculate mean NDVI
results = []
for idx, zone in zones.iterrows():
    # Clip to zone
    clipped = ds.rio.clip([zone.geometry], zones.crs)
    
    # Calculate mean
    mean_ndvi = clipped.NDVI.mean(dim=['x', 'y']).values
    
    results.append({
        'zone_id': zone['id'],
        'zone_name': zone['name'],
        'mean_ndvi': mean_ndvi
    })

# Convert to DataFrame
import pandas as pd
stats_df = pd.DataFrame(results)
print(stats_df)
```

### Grid-based Aggregation

Aggregate to coarser resolution:

```python
# Coarsen by factor of 10
coarse = ds.coarsen(x=10, y=10, boundary='trim').mean()

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ds.NDVI.isel(time=0).plot(ax=axes[0])
axes[0].set_title('Original (10m)')

coarse.NDVI.isel(time=0).plot(ax=axes[1])
axes[1].set_title('Aggregated (100m)')

plt.tight_layout()
plt.show()
```

## Temporal Aggregation

### Composites

Create temporal composites:

```python
# Monthly median composite
monthly_composite = ds.resample(time='1M').median()

# Annual maximum composite
annual_max = ds.resample(time='1Y').max()

# Seasonal composites
seasonal = ds.groupby('time.season').mean()

# Visualize seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
seasons = ['DJF', 'MAM', 'JJA', 'SON']

for idx, season in enumerate(seasons):
    ax = axes[idx // 2, idx % 2]
    if season in seasonal.season:
        seasonal.sel(season=season).NDVI.plot(ax=ax, cmap='RdYlGn')
        ax.set_title(f'{season} Mean NDVI')
        ax.set_axis_off()

plt.tight_layout()
plt.show()
```

## GroupBy Operations

### Group by Categories

```python
# Group by month
monthly_stats = ds.groupby('time.month').mean()

# Plot monthly climatology
monthly_stats.NDVI.mean(dim=['x', 'y']).plot(marker='o')
plt.title('Monthly NDVI Climatology')
plt.xlabel('Month')
plt.ylabel('Mean NDVI')
plt.grid(True, alpha=0.3)
plt.show()
```

### Custom Grouping

```python
# Define seasons manually
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Apply custom grouping
seasons = xr.DataArray(
    [get_season(m) for m in ds.time.dt.month.values],
    dims='time',
    coords={'time': ds.time}
)

seasonal_custom = ds.groupby(seasons).mean()
```

## Combined Aggregation

### Spatio-temporal Aggregation

```python
# Calculate regional monthly means
regional_monthly = ds.sel(
    x=slice(82.0, 83.0),
    y=slice(26.5, 27.5)
).resample(time='1M').mean().mean(dim=['x', 'y'])

# Plot
regional_monthly.NDVI.plot(marker='o')
plt.title('Regional Monthly Mean NDVI')
plt.ylabel('NDVI')
plt.grid(True, alpha=0.3)
plt.show()
```

## Performance Tips

### Use Dask for Large Datasets

```python
from dask.distributed import Client

client = Client()

# Load with chunks
ds = xr.open_zarr('large_data.zarr', chunks={'time': 10, 'x': 512, 'y': 512})

# Aggregate (lazy)
monthly = ds.resample(time='1M').mean()

# Compute
result = monthly.compute()
```

## Next Steps

→ Continue to [Advanced Topics](../advanced/scaling-dask.md)

## Additional Resources

- [XArray GroupBy](https://docs.xarray.dev/en/stable/user-guide/groupby.html)
- [Rasterio Zonal Stats](https://pythonhosted.org/rasterstats/)
