# Water Quality Monitoring with XEE

Monitor water quality parameters using satellite imagery and Earth Engine data accessed through XEE.

## Overview

This example demonstrates:

- Water body detection and masking
- Turbidity estimation
- Chlorophyll-a concentration
- Temporal water quality trends
- Multi-lake comparison

**Dataset**: Sentinel-2 for water quality parameters

## Step 1: Initialize and Define Study Area

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define lake region (example: Dal Lake, Kashmir)
lake_center = ee.Geometry.Point([74.8723, 34.1083])
lake_roi = lake_center.buffer(2000)  # 2km buffer

# Time period
start_date = '2023-01-01'
end_date = '2023-12-31'

print("Study area defined")
```

## Step 2: Load and Process Sentinel-2 Data

```python
def mask_s2_clouds(image):
    """Mask clouds and cirrus."""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Load Sentinel-2 collection
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(lake_roi) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(mask_s2_clouds)

print(f"Found {collection.size().getInfo()} images")

# Create median composite
composite = collection.median().clip(lake_roi)

# Load with XEE
ds = xr.open_dataset(
    composite,
    engine='ee',
    geometry=lake_roi,
    scale=20,
    crs='EPSG:4326'
).compute()

print("Data loaded")
```

## Step 3: Water Body Detection

```python
# Calculate NDWI (Normalized Difference Water Index)
ndwi = (ds.B3 - ds.B8) / (ds.B3 + ds.B8)

# Calculate MNDWI (Modified NDWI)
mndwi = (ds.B3 - ds.B11) / (ds.B3 + ds.B11)

# Water mask (MNDWI > 0 indicates water)
water_mask = mndwi > 0

# Apply water mask to dataset
ds_water = ds.where(water_mask)

# Visualize water detection
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RGB
rgb = np.stack([
    ds.B4.values / 3000,
    ds.B3.values / 3000,
    ds.B2.values / 3000
], axis=-1)
rgb = np.clip(rgb, 0, 1)
axes[0].imshow(rgb)
axes[0].set_title('True Color')
axes[0].axis('off')

# MNDWI
mndwi.plot(ax=axes[1], cmap='RdYlBu', vmin=-1, vmax=1)
axes[1].set_title('MNDWI (Water Index)')
axes[1].set_axis_off()

# Water mask
water_mask.plot(ax=axes[2], cmap='Blues')
axes[2].set_title('Water Mask')
axes[2].set_axis_off()

plt.tight_layout()
plt.show()
```

## Step 4: Calculate Water Quality Parameters

```python
# Turbidity estimation using Red/Blue ratio
# Higher values indicate more turbidity
turbidity_index = ds_water.B4 / ds_water.B2

# Chlorophyll-a estimation using band ratios
# Based on OC2 algorithm (simplified)
chl_a = ds_water.B3 / ds_water.B4

# Suspended sediment index
ssi = ds_water.B4 + ds_water.B3

# Total Suspended Matter (TSM) - empirical relationship
tsm = 3.0 * ds_water.B4 - 0.5  # Simplified model

print("Water quality parameters calculated")
```

## Step 5: Visualize Water Quality

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Turbidity
im1 = turbidity_index.plot(ax=axes[0, 0], cmap='YlOrBr', vmin=0, vmax=2)
axes[0, 0].set_title('Turbidity Index (Red/Blue Ratio)')
axes[0, 0].set_axis_off()

# Chlorophyll-a
im2 = chl_a.plot(ax=axes[0, 1], cmap='YlGn', vmin=0, vmax=3)
axes[0, 1].set_title('Chlorophyll-a Proxy')
axes[0, 1].set_axis_off()

# Suspended Sediment
im3 = ssi.plot(ax=axes[1, 0], cmap='copper', vmin=0, vmax=2000)
axes[1, 0].set_title('Suspended Sediment Index')
axes[1, 0].set_axis_off()

# TSM
im4 = tsm.plot(ax=axes[1, 1], cmap='RdYlBu_r', vmin=0, vmax=100)
axes[1, 1].set_title('Total Suspended Matter (mg/L)')
axes[1, 1].set_axis_off()

plt.tight_layout()
plt.show()
```

## Step 6: Temporal Analysis

```python
# Load time series for temporal analysis
def calculate_water_quality(image):
    """Calculate water quality for a single image."""
    # Mask water
    mndwi = image.normalizedDifference(['B3', 'B11'])
    water = image.updateMask(mndwi.gt(0))
    
    # Calculate parameters
    turbidity = water.select('B4').divide(water.select('B2'))
    chl_a = water.select('B3').divide(water.select('B4'))
    
    return image.addBands([
        turbidity.rename('turbidity'),
        chl_a.rename('chl_a')
    ])

# Apply to collection
wq_collection = collection.map(calculate_water_quality)

# Load time series
ds_ts = xr.open_dataset(
    wq_collection,
    engine='ee',
    geometry=lake_roi,
    scale=100,  # Lower resolution for faster processing
    crs='EPSG:4326'
)

ds_ts = ds_ts.sortby('time').compute()

# Calculate mean values over lake
turbidity_ts = ds_ts.turbidity.mean(dim=['lon', 'lat'])
chl_a_ts = ds_ts.chl_a.mean(dim=['lon', 'lat'])

# Plot time series
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Turbidity over time
turbidity_ts.plot(ax=axes[0], marker='o', linestyle='-', color='brown')
axes[0].set_title('Lake Turbidity Over Time')
axes[0].set_ylabel('Turbidity Index')
axes[0].grid(True, alpha=0.3)

# Chlorophyll-a over time
chl_a_ts.plot(ax=axes[1], marker='o', linestyle='-', color='green')
axes[1].set_title('Chlorophyll-a Proxy Over Time')
axes[1].set_ylabel('Chl-a Index')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 7: Water Quality Classification

```python
# Classify water quality based on turbidity and chlorophyll
def classify_water_quality(turbidity, chl_a):
    """Classify water quality into categories."""
    quality = np.zeros_like(turbidity.values)
    
    # Good: Low turbidity, low chlorophyll
    good = (turbidity < 1.0) & (chl_a < 1.5)
    quality[good.values] = 1
    
    # Moderate: Medium turbidity or chlorophyll
    moderate = ((turbidity >= 1.0) & (turbidity < 1.5)) | \
               ((chl_a >= 1.5) & (chl_a < 2.0))
    quality[moderate.values] = 2
    
    # Poor: High turbidity or chlorophyll
    poor = (turbidity >= 1.5) | (chl_a >= 2.0)
    quality[poor.values] = 3
    
    return quality

# Classify
wq_class = classify_water_quality(turbidity_index, chl_a)

# Visualize classification
from matplotlib.colors import ListedColormap

colors = ['white', 'green', 'yellow', 'red']
labels = ['No Data', 'Good', 'Moderate', 'Poor']
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(wq_class, cmap=cmap, vmin=0, vmax=3)
ax.set_title('Water Quality Classification')
ax.set_axis_off()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) 
                   for i in range(len(labels))]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Calculate statistics
total_pixels = np.sum(wq_class > 0)
for i, label in enumerate(labels[1:], 1):
    count = np.sum(wq_class == i)
    percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
    print(f"{label}: {percentage:.1f}%")
```

## Step 8: Multi-Lake Comparison

```python
# Define multiple lakes
lakes = [
    {'name': 'Lake A', 'lon': 74.87, 'lat': 34.11, 'buffer': 2000},
    {'name': 'Lake B', 'lon': 74.90, 'lat': 34.08, 'buffer': 1500},
    {'name': 'Lake C', 'lon': 74.85, 'lat': 34.13, 'buffer': 1000}
]

# Compare water quality across lakes
comparison_results = []

for lake in lakes:
    roi = ee.Geometry.Point([lake['lon'], lake['lat']]).buffer(lake['buffer'])
    
    # Get composite
    lake_composite = collection.median().clip(roi)
    
    # Load data
    lake_ds = xr.open_dataset(
        lake_composite,
        engine='ee',
        geometry=roi,
        scale=50,
        crs='EPSG:4326'
    ).compute()
    
    # Calculate indices
    lake_mndwi = (lake_ds.B3 - lake_ds.B11) / (lake_ds.B3 + lake_ds.B11)
    lake_water = lake_ds.where(lake_mndwi > 0)
    
    lake_turbidity = (lake_water.B4 / lake_water.B2).mean().values.item()
    lake_chl_a = (lake_water.B3 / lake_water.B4).mean().values.item()
    
    comparison_results.append({
        'Lake': lake['name'],
        'Turbidity': lake_turbidity,
        'Chl-a': lake_chl_a
    })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)
print("\nLake Comparison:")
print(comparison_df)

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Turbidity comparison
axes[0].bar(comparison_df['Lake'], comparison_df['Turbidity'], color='brown', alpha=0.7)
axes[0].set_ylabel('Turbidity Index')
axes[0].set_title('Turbidity Comparison')
axes[0].grid(True, alpha=0.3, axis='y')

# Chlorophyll comparison
axes[1].bar(comparison_df['Lake'], comparison_df['Chl-a'], color='green', alpha=0.7)
axes[1].set_ylabel('Chl-a Index')
axes[1].set_title('Chlorophyll-a Comparison')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## Step 9: Export Results

```python
# Create comprehensive water quality dataset
wq_results = xr.Dataset({
    'turbidity': (['y', 'x'], turbidity_index.values),
    'chl_a': (['y', 'x'], chl_a.values),
    'tsm': (['y', 'x'], tsm.values),
    'water_mask': (['y', 'x'], water_mask.values.astype(int)),
    'quality_class': (['y', 'x'], wq_class)
}, coords={
    'y': ds.lat.values,
    'x': ds.lon.values
})

# Add CRS
wq_results = wq_results.rio.write_crs('EPSG:4326')

# Save as NetCDF
wq_results.to_netcdf('water_quality_results.nc')
print("Results saved to water_quality_results.nc")

# Export individual parameters as GeoTIFF
turbidity_da = xr.DataArray(
    turbidity_index.values,
    dims=['y', 'x'],
    coords={'y': ds.lat.values, 'x': ds.lon.values}
)
turbidity_da = turbidity_da.rio.write_crs('EPSG:4326')
turbidity_da.rio.to_raster('turbidity.tif')

# Save time series
ts_df = pd.DataFrame({
    'date': pd.to_datetime(ds_ts.time.values),
    'turbidity': turbidity_ts.values,
    'chl_a': chl_a_ts.values
})
ts_df.to_csv('water_quality_timeseries.csv', index=False)

# Save comparison
comparison_df.to_csv('lake_comparison.csv', index=False)

print("All results exported successfully")
```

## Key Takeaways

!!! success "What You Learned"
    - Water body detection using MNDWI
    - Turbidity estimation from band ratios
    - Chlorophyll-a proxy calculation
    - Total Suspended Matter estimation
    - Temporal water quality monitoring
    - Water quality classification
    - Multi-lake comparison
    - Comprehensive result export

## Additional Resources

- [Water Quality Remote Sensing](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/water-quality)
- [Sentinel-2 for Water Quality](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/water_quality/)
- [Earth Engine Water Detection](https://developers.google.com/earth-engine/tutorials/community/water-detection)
