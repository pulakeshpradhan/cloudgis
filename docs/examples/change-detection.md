# Change Detection with XEE

Detect and quantify land surface changes using multi-temporal Earth Engine data accessed through XEE.

## Overview

This example demonstrates:

- Pre/post event comparison
- Change magnitude calculation
- Statistical significance testing
- Change visualization and mapping

**Scenario**: Detecting urban expansion between 2020 and 2023

## Step 1: Initialize and Define Parameters

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define region of interest
roi = ee.Geometry.Rectangle([77.1, 28.5, 77.3, 28.7])  # Delhi suburbs

# Define time periods
period_before = ('2020-01-01', '2020-03-31')
period_after = ('2023-01-01', '2023-03-31')
```

## Step 2: Load Data for Both Periods

```python
def get_composite(start_date, end_date, roi):
    """Create cloud-free composite for a time period."""
    
    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                     qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .map(mask_clouds)
    
    return collection.median().clip(roi)

# Create composites
composite_before = get_composite(*period_before, roi)
composite_after = get_composite(*period_after, roi)

print("Composites created")
```

## Step 3: Load with XEE

```python
# Load both periods
ds_before = xr.open_dataset(
    composite_before,
    engine='ee',
    geometry=roi,
    scale=20,
    crs='EPSG:4326'
).compute()

ds_after = xr.open_dataset(
    composite_after,
    engine='ee',
    geometry=roi,
    scale=20,
    crs='EPSG:4326'
).compute()

print("Data loaded")
print(f"Shape: {ds_before.B4.shape}")
```

## Step 4: Calculate Indices for Both Periods

```python
# NDVI (vegetation)
ndvi_before = (ds_before.B8 - ds_before.B4) / (ds_before.B8 + ds_before.B4)
ndvi_after = (ds_after.B8 - ds_after.B4) / (ds_after.B8 + ds_after.B4)

# NDBI (built-up)
ndbi_before = (ds_before.B11 - ds_before.B8) / (ds_before.B11 + ds_before.B8)
ndbi_after = (ds_after.B11 - ds_after.B8) / (ds_after.B11 + ds_after.B8)

# NDWI (water)
ndwi_before = (ds_before.B3 - ds_before.B8) / (ds_before.B3 + ds_before.B8)
ndwi_after = (ds_after.B3 - ds_after.B8) / (ds_after.B3 + ds_after.B8)

print("Indices calculated")
```

## Step 5: Calculate Change

```python
# Calculate differences
ndvi_change = ndvi_after - ndvi_before
ndbi_change = ndbi_after - ndbi_before
ndwi_change = ndwi_after - ndwi_before

# Calculate percentage change
ndvi_pct_change = ((ndvi_after - ndvi_before) / np.abs(ndvi_before)) * 100
ndbi_pct_change = ((ndbi_after - ndbi_before) / np.abs(ndbi_before)) * 100

print("Change calculated")
```

## Step 6: Visualize Before/After

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# NDVI Before
ndvi_before.plot(ax=axes[0, 0], cmap='RdYlGn', vmin=-1, vmax=1)
axes[0, 0].set_title('NDVI - 2020')
axes[0, 0].set_axis_off()

# NDVI After
ndvi_after.plot(ax=axes[0, 1], cmap='RdYlGn', vmin=-1, vmax=1)
axes[0, 1].set_title('NDVI - 2023')
axes[0, 1].set_axis_off()

# NDVI Change
im1 = ndvi_change.plot(ax=axes[0, 2], cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[0, 2].set_title('NDVI Change (2023-2020)')
axes[0, 2].set_axis_off()

# NDBI Before
ndbi_before.plot(ax=axes[1, 0], cmap='YlOrRd', vmin=-1, vmax=1)
axes[1, 0].set_title('NDBI - 2020')
axes[1, 0].set_axis_off()

# NDBI After
ndbi_after.plot(ax=axes[1, 1], cmap='YlOrRd', vmin=-1, vmax=1)
axes[1, 1].set_title('NDBI - 2023')
axes[1, 1].set_axis_off()

# NDBI Change
im2 = ndbi_change.plot(ax=axes[1, 2], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1, 2].set_title('NDBI Change (Urban Expansion)')
axes[1, 2].set_axis_off()

plt.tight_layout()
plt.show()
```

## Step 7: Detect Significant Changes

```python
# Define thresholds for significant change
ndvi_threshold = 0.2  # Decrease indicates vegetation loss
ndbi_threshold = 0.2  # Increase indicates urbanization

# Detect changes
vegetation_loss = ndvi_change < -ndvi_threshold
urban_expansion = ndbi_change > ndbi_threshold
water_change = np.abs(ndwi_change) > 0.2

# Combine into change categories
change_map = np.zeros_like(ndvi_change.values)
change_map[vegetation_loss.values] = 1  # Vegetation loss
change_map[urban_expansion.values] = 2  # Urban expansion
change_map[water_change.values] = 3     # Water change

# Visualize change categories
from matplotlib.colors import ListedColormap

colors = ['white', 'red', 'gray', 'blue']
labels = ['No Change', 'Vegetation Loss', 'Urban Expansion', 'Water Change']
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(change_map, cmap=cmap, vmin=0, vmax=3)
ax.set_title('Change Detection Map (2020-2023)', fontsize=14)
ax.set_axis_off()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) 
                   for i in range(len(labels))]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()
```

## Step 8: Calculate Change Statistics

```python
# Calculate areas
pixel_area_sqm = 20 * 20  # 20m x 20m
pixel_area_sqkm = pixel_area_sqm / 1e6

# Count pixels for each change type
stats = {}
for i, label in enumerate(labels):
    pixel_count = np.sum(change_map == i)
    area_sqkm = pixel_count * pixel_area_sqkm
    percentage = (pixel_count / change_map.size) * 100
    
    stats[label] = {
        'pixels': pixel_count,
        'area_km2': area_sqkm,
        'percentage': percentage
    }
    
    print(f"{label}:")
    print(f"  Area: {area_sqkm:.2f} km²")
    print(f"  Percentage: {percentage:.1f}%")
    print()

# Visualize statistics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
areas = [stats[label]['area_km2'] for label in labels]
axes[0].bar(labels, areas, color=colors)
axes[0].set_ylabel('Area (km²)')
axes[0].set_title('Change Areas')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart (excluding no change)
change_labels = labels[1:]
change_areas = areas[1:]
axes[1].pie(change_areas, labels=change_labels, colors=colors[1:],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Distribution of Changes')

plt.tight_layout()
plt.show()
```

## Step 9: Temporal Profile Analysis

```python
# Extract temporal profiles at specific points
points_of_interest = [
    {'name': 'Urban Expansion Site', 'lon': 77.15, 'lat': 28.60},
    {'name': 'Stable Vegetation', 'lon': 77.25, 'lat': 28.65},
    {'name': 'Water Body', 'lon': 77.20, 'lat': 28.55}
]

fig, axes = plt.subplots(len(points_of_interest), 1, figsize=(12, 10))

for idx, poi in enumerate(points_of_interest):
    # Extract values at point
    ndvi_b = ndvi_before.sel(lon=poi['lon'], lat=poi['lat'], method='nearest').values
    ndvi_a = ndvi_after.sel(lon=poi['lon'], lat=poi['lat'], method='nearest').values
    
    ndbi_b = ndbi_before.sel(lon=poi['lon'], lat=poi['lat'], method='nearest').values
    ndbi_a = ndbi_after.sel(lon=poi['lon'], lat=poi['lat'], method='nearest').values
    
    # Plot
    x = ['2020', '2023']
    axes[idx].plot(x, [ndvi_b, ndvi_a], 'o-', label='NDVI', linewidth=2, markersize=8)
    axes[idx].plot(x, [ndbi_b, ndbi_a], 's-', label='NDBI', linewidth=2, markersize=8)
    axes[idx].set_title(poi['name'])
    axes[idx].set_ylabel('Index Value')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 10: Export Results

```python
# Create change detection dataset
change_ds = xr.Dataset({
    'ndvi_change': (['y', 'x'], ndvi_change.values),
    'ndbi_change': (['y', 'x'], ndbi_change.values),
    'change_category': (['y', 'x'], change_map)
}, coords={
    'y': ds_before.lat.values,
    'x': ds_before.lon.values
})

# Add CRS
change_ds = change_ds.rio.write_crs('EPSG:4326')

# Save as NetCDF
change_ds.to_netcdf('change_detection_results.nc')
print("Results saved to change_detection_results.nc")

# Export change map as GeoTIFF
change_da = xr.DataArray(
    change_map,
    dims=['y', 'x'],
    coords={'y': ds_before.lat.values, 'x': ds_before.lon.values}
)
change_da = change_da.rio.write_crs('EPSG:4326')
change_da.rio.to_raster('change_map.tif')
print("Change map saved to change_map.tif")

# Export statistics
import pandas as pd
stats_df = pd.DataFrame(stats).T
stats_df.to_csv('change_statistics.csv')
print("Statistics saved to change_statistics.csv")
```

## Advanced: Change Trajectory Analysis

```python
# For more detailed analysis, load monthly data
def get_monthly_ndvi(year, roi):
    """Get monthly NDVI composites for a year."""
    monthly_ndvi = []
    
    for month in range(1, 13):
        start = f'{year}-{month:02d}-01'
        if month == 12:
            end = f'{year+1}-01-01'
        else:
            end = f'{year}-{month+1:02d}-01'
        
        composite = get_composite(start, end, roi)
        ds = xr.open_dataset(composite, engine='ee', geometry=roi, scale=100, crs='EPSG:4326').compute()
        ndvi = (ds.B8 - ds.B4) / (ds.B8 + ds.B4)
        monthly_ndvi.append(ndvi.mean().values)
    
    return monthly_ndvi

# Get trajectories for both years
ndvi_2020 = get_monthly_ndvi(2020, roi)
ndvi_2023 = get_monthly_ndvi(2023, roi)

# Plot trajectory comparison
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 6))
plt.plot(months, ndvi_2020, 'o-', label='2020', linewidth=2, markersize=8)
plt.plot(months, ndvi_2023, 's-', label='2023', linewidth=2, markersize=8)
plt.xlabel('Month')
plt.ylabel('Mean NDVI')
plt.title('Annual NDVI Trajectory Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Key Takeaways

!!! success "What You Learned"
    - Creating multi-temporal composites with XEE
    - Calculating change in multiple indices
    - Detecting significant changes with thresholds
    - Categorizing different types of change
    - Visualizing before/after comparisons
    - Calculating change statistics and areas
    - Analyzing temporal trajectories
    - Exporting change detection results

## Next Steps

→ Continue to [Time Series and Phenological Methods](timeseries-phenology.md)

## Additional Resources

- [Change Detection Methods](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/change-detection)
- [Earth Engine Change Detection](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1)
