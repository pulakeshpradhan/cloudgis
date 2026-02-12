# Cloud Masking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spatialthoughts/courses/blob/master/code/python_remote_sensing/04_cloud_masking.ipynb)

## Overview

Clouds are a major challenge in optical remote sensing. They obscure the Earth's surface and can introduce errors in analysis. Proper cloud masking is essential for accurate results.

In this section, we'll learn how to:

- Use quality assessment (QA) bands for cloud detection
- Create cloud masks using bit manipulation
- Apply masks to remove cloudy pixels
- Combine cloud and shadow masks
- Handle partially cloudy scenes

## Understanding QA Bands

Most satellite products include Quality Assessment (QA) bands that contain pixel-level quality information encoded as bit flags.

### Sentinel-2 QA60 Band

The QA60 band contains cloud information:

- Bit 10: Opaque clouds
- Bit 11: Cirrus clouds

### Landsat QA_PIXEL Band

Contains multiple quality flags:

- Bit 3: Cloud
- Bit 4: Cloud shadow
- Bit 1: Dilated cloud

## Bit Manipulation

Quality flags are stored as binary values. We use bitwise operations to extract specific flags.

### Binary Representation

```python
# Example: Number 1024 in binary
binary = bin(1024)
print(binary)  # '0b10000000000'

# Bit 10 is set (counting from right, starting at 0)
```

### Bitwise AND Operation

```python
# Check if bit 10 is set
value = 1024
bit_10_mask = 1 << 10  # Shift 1 left by 10 positions = 1024
result = value & bit_10_mask
print(result)  # 1024 (bit is set)

# Check if bit 11 is set
bit_11_mask = 1 << 11  # 2048
result = value & bit_11_mask
print(result)  # 0 (bit is not set)
```

## Setup

```python
%%capture
if 'google.colab' in str(get_ipython()):
    !pip install pystac-client odc-stac rioxarray dask
```

```python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client
from odc import stac
import xarray as xr
import rioxarray as rxr
```

## Get Sentinel-2 Scene

```python
latitude = 27.163
longitude = 82.608
year = 2023

# Define bounding box
km2deg = 1.0 / 111
x, y = (longitude, latitude)
r = 1 * km2deg
bbox = (x - r, y - r, x + r, y + r)

# Query STAC Catalog
catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1')

# Search for cloudy scenes to demonstrate masking
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime=f'{year}',
    query={
        'eo:cloud_cover': {'gt': 20, 'lt': 60}  # Moderately cloudy
    },
    sortby=[{
        'field': 'properties.eo:cloud_cover',
        'direction': 'asc'
    }]
)
items = search.item_collection()

# Load data including QA band
ds = stac.load(
    items,
    bands=['red', 'green', 'blue', 'nir', 'qa60'],  # Include QA band
    resolution=20,
    chunks={},
    groupby='solar_day',
    preserve_original_order=True
)

# Select a scene
timestamp = pd.to_datetime(items[0].properties['datetime']).tz_convert(None)
scene = ds.sel(time=timestamp).compute()

# Apply scale and offset (not needed for QA band)
scale = 0.0001
offset = -0.1
scene_scaled = scene[['red', 'green', 'blue', 'nir']].where(
    scene[['red', 'green', 'blue', 'nir']] != 0) * scale + offset

# Add QA band back
scene_scaled['qa60'] = scene.qa60
scene = scene_scaled
```

## Visualize the Scene

```python
scene_da = scene[['red', 'green', 'blue']].to_array('band')

fig, ax = plt.subplots(figsize=(10, 8))
scene_da.plot.imshow(ax=ax, robust=True)
ax.set_title('Original Scene (with clouds)')
ax.set_axis_off()
plt.show()
```

## Create a Cloud Mask

### Extract Cloud Bits

```python
# Get QA band
qa = scene.qa60

# Create masks for opaque and cirrus clouds
opaque_clouds = (qa & (1 << 10)) != 0
cirrus_clouds = (qa & (1 << 11)) != 0

# Combine cloud masks
cloud_mask = opaque_clouds | cirrus_clouds

# Visualize cloud mask
fig, ax = plt.subplots(figsize=(10, 8))
cloud_mask.plot(ax=ax, cmap='RdYlGn_r', add_colorbar=False)
ax.set_title('Cloud Mask (True = Cloud)')
ax.set_axis_off()
plt.show()
```

### Apply Cloud Mask

```python
# Invert mask (True = clear, False = cloud)
clear_mask = ~cloud_mask

# Apply mask to scene
scene_masked = scene.where(clear_mask)

# Visualize masked scene
scene_masked_da = scene_masked[['red', 'green', 'blue']].to_array('band')

fig, ax = plt.subplots(figsize=(10, 8))
scene_masked_da.plot.imshow(ax=ax, robust=True)
ax.set_title('Cloud-Masked Scene')
ax.set_axis_off()
plt.show()
```

## Before/After Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
scene_da.plot.imshow(ax=axes[0], robust=True)
axes[0].set_title('Original (with clouds)')
axes[0].set_axis_off()

# Cloud mask
cloud_mask.plot(ax=axes[1], cmap='RdYlGn_r', add_colorbar=False)
axes[1].set_title('Cloud Mask')
axes[1].set_axis_off()

# Masked
scene_masked_da.plot.imshow(ax=axes[2], robust=True)
axes[2].set_title('Cloud-Masked')
axes[2].set_axis_off()

plt.tight_layout()
plt.show()
```

## Cloud Statistics

```python
# Calculate cloud coverage
total_pixels = cloud_mask.size
cloud_pixels = cloud_mask.sum().values
cloud_percentage = (cloud_pixels / total_pixels) * 100

print(f"Total pixels: {total_pixels:,}")
print(f"Cloud pixels: {cloud_pixels:,}")
print(f"Cloud coverage: {cloud_percentage:.2f}%")

# Compare with metadata
metadata_cloud_cover = items[0].properties.get('eo:cloud_cover', 'N/A')
print(f"Metadata cloud cover: {metadata_cloud_cover}%")
```

## Advanced Cloud Masking

### Morphological Operations

Dilate cloud mask to include cloud edges:

```python
from scipy import ndimage

# Convert to numpy array
cloud_array = cloud_mask.values

# Dilate cloud mask (expand clouds)
structure = np.ones((5, 5))  # 5x5 kernel
dilated_clouds = ndimage.binary_dilation(cloud_array, structure=structure)

# Convert back to DataArray
dilated_mask = xr.DataArray(
    dilated_clouds,
    coords=cloud_mask.coords,
    dims=cloud_mask.dims
)

# Apply dilated mask
scene_dilated_masked = scene.where(~dilated_mask)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cloud_mask.plot(ax=axes[0], cmap='RdYlGn_r', add_colorbar=False)
axes[0].set_title('Original Cloud Mask')
axes[0].set_axis_off()

dilated_mask.plot(ax=axes[1], cmap='RdYlGn_r', add_colorbar=False)
axes[1].set_title('Dilated Cloud Mask')
axes[1].set_axis_off()

plt.tight_layout()
plt.show()
```

### Cloud Shadow Detection

Estimate cloud shadows using geometry:

```python
# Simple cloud shadow detection using NIR threshold
nir_threshold = 0.15
potential_shadows = scene.nir < nir_threshold

# Combine with cloud mask (shadows are near clouds)
# Shift cloud mask to approximate shadow location
shadow_offset = 10  # pixels
cloud_shifted = np.roll(cloud_array, shadow_offset, axis=0)
cloud_shifted = np.roll(cloud_shifted, shadow_offset, axis=1)

# Shadows are dark areas near clouds
shadow_mask = potential_shadows.values & cloud_shifted & ~cloud_array

# Convert to DataArray
shadow_mask_da = xr.DataArray(
    shadow_mask,
    coords=cloud_mask.coords,
    dims=cloud_mask.dims
)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
shadow_mask_da.plot(ax=ax, cmap='gray', add_colorbar=False)
ax.set_title('Estimated Cloud Shadows')
ax.set_axis_off()
plt.show()
```

### Combined Cloud and Shadow Mask

```python
# Combine cloud and shadow masks
combined_mask = cloud_mask | shadow_mask_da

# Apply combined mask
scene_fully_masked = scene.where(~combined_mask)

# Visualize
scene_fully_masked_da = scene_fully_masked[['red', 'green', 'blue']].to_array('band')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scene_masked_da.plot.imshow(ax=axes[0], robust=True)
axes[0].set_title('Cloud-Masked Only')
axes[0].set_axis_off()

scene_fully_masked_da.plot.imshow(ax=axes[1], robust=True)
axes[1].set_title('Cloud and Shadow Masked')
axes[1].set_axis_off()

plt.tight_layout()
plt.show()
```

## Time Series Cloud Masking

```python
# Load full time series
ds_full = stac.load(
    items,
    bands=['red', 'nir', 'qa60'],
    resolution=20,
    chunks={'time': 10}
).compute()

# Apply scale/offset
ds_scaled = ds_full[['red', 'nir']].where(ds_full[['red', 'nir']] != 0) * scale + offset
ds_scaled['qa60'] = ds_full.qa60

# Create cloud masks for all time steps
qa_full = ds_scaled.qa60
cloud_masks = (qa_full & (1 << 10)) != 0 | (qa_full & (1 << 11)) != 0

# Apply masks
ds_masked = ds_scaled.where(~cloud_masks)

# Calculate NDVI
ndvi = (ds_masked.nir - ds_masked.red) / (ds_masked.nir + ds_masked.red)

# Calculate temporal mean (ignoring masked pixels)
ndvi_mean = ndvi.mean(dim='time')

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
ndvi_mean.plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_title('Mean NDVI (Cloud-Masked Time Series)')
ax.set_axis_off()
plt.show()
```

## Quality Metrics

```python
# Calculate valid pixel percentage for each time step
valid_pixels = (~cloud_masks).sum(dim=['x', 'y'])
total_pixels = cloud_masks.sizes['x'] * cloud_masks.sizes['y']
valid_percentage = (valid_pixels / total_pixels) * 100

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
valid_percentage.plot(ax=ax, marker='o')
ax.set_title('Valid (Cloud-Free) Pixel Percentage Over Time')
ax.set_ylabel('Valid Pixels (%)')
ax.grid(True, alpha=0.3)
ax.axhline(y=50, color='r', linestyle='--', label='50% threshold')
ax.legend()
plt.show()
```

## Exercise

Create a function that takes a Sentinel-2 scene and returns a cloud-free composite by:

1. Creating a cloud mask
2. Dilating the mask by 3 pixels
3. Applying the mask to the scene
4. Calculating the percentage of valid pixels

**Solution:**

```python
def create_cloud_free_scene(scene, dilation_size=3):
    """
    Create cloud-free scene from Sentinel-2 data.
    
    Parameters:
    -----------
    scene : xarray.Dataset
        Sentinel-2 scene with qa60 band
    dilation_size : int
        Size of dilation kernel
        
    Returns:
    --------
    masked_scene : xarray.Dataset
        Cloud-masked scene
    valid_percentage : float
        Percentage of valid pixels
    """
    # Extract QA band
    qa = scene.qa60
    
    # Create cloud mask
    cloud_mask = (qa & (1 << 10)) != 0 | (qa & (1 << 11)) != 0
    
    # Dilate mask
    structure = np.ones((dilation_size, dilation_size))
    dilated = ndimage.binary_dilation(cloud_mask.values, structure=structure)
    dilated_mask = xr.DataArray(dilated, coords=cloud_mask.coords, dims=cloud_mask.dims)
    
    # Apply mask
    masked_scene = scene.where(~dilated_mask)
    
    # Calculate valid percentage
    valid_pixels = (~dilated_mask).sum().values
    total_pixels = dilated_mask.size
    valid_percentage = (valid_pixels / total_pixels) * 100
    
    return masked_scene, valid_percentage

# Use function
masked, valid_pct = create_cloud_free_scene(scene, dilation_size=3)
print(f"Valid pixels: {valid_pct:.2f}%")
```

## Key Takeaways

!!! success "What You Learned"
    - QA bands contain pixel-level quality information
    - Bitwise operations extract specific quality flags
    - Cloud masks identify and remove cloudy pixels
    - Morphological operations refine masks
    - Cloud shadows can be detected and masked
    - Time series masking improves composite quality
    - Valid pixel percentage indicates data quality

## Next Steps

→ Continue to [Time Series Extraction](time-series.md)

## Additional Resources

- [Sentinel-2 Cloud Masks](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm)
- [Landsat QA Bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)
- [Cloud Masking Algorithms](https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/cloud_detector)
- [S2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector)
