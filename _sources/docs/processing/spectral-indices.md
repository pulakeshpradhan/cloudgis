# Calculating Spectral Indices

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spatialthoughts/courses/blob/master/code/python_remote_sensing/03_calculating_indices.ipynb)

## Overview

Spectral indices are core to many remote sensing analyses. They combine different spectral bands to highlight specific features or properties of the Earth's surface.

In this section, we'll learn how to:

- Calculate common spectral indices (NDVI, NDWI, SAVI)
- Visualize index results
- Save computed indices
- Create custom indices

## Common Spectral Indices

### NDVI (Normalized Difference Vegetation Index)

Measures vegetation health and density:

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

- **Range**: -1 to +1
- **Interpretation**:
  - < 0: Water, snow, clouds
  - 0-0.2: Bare soil, rock
  - 0.2-0.5: Sparse vegetation
  - 0.5-0.8: Moderate vegetation
  - > 0.8: Dense vegetation

### NDWI (Normalized Difference Water Index)

Detects water bodies and moisture:

$$NDWI = \frac{Green - NIR}{Green + NIR}$$

- **Range**: -1 to +1
- **Interpretation**:
  - > 0: Water
  - < 0: Non-water

### MNDWI (Modified NDWI)

Better for urban areas:

$$MNDWI = \frac{Green - SWIR}{Green + SWIR}$$

### SAVI (Soil Adjusted Vegetation Index)

Minimizes soil brightness influence:

$$SAVI = \frac{(NIR - Red) \times (1 + L)}{NIR + Red + L}$$

Where L = 0.5 (soil brightness correction factor)

### EVI (Enhanced Vegetation Index)

Improved sensitivity in high biomass regions:

$$EVI = 2.5 \times \frac{NIR - Red}{NIR + 6 \times Red - 7.5 \times Blue + 1}$$

### NDBI (Normalized Difference Built-up Index)

Identifies built-up areas:

$$NDBI = \frac{SWIR - NIR}{SWIR + NIR}$$

## Setup

```python
%%capture
if 'google.colab' in str(get_ipython()):
    !pip install pystac-client odc-stac rioxarray dask jupyter-server-proxy
```

```python
import os
import matplotlib.pyplot as plt
import pandas as pd
import pystac_client
from odc import stac
import xarray as xr
import rioxarray as rxr
from dask.distributed import Client

client = Client()
client
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

search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime=f'{year}',
    query={
        'eo:cloud_cover': {'lt': 30},
        's2:nodata_pixel_percentage': {'lt': 10}
    },
    sortby=[{
        'field': 'properties.eo:cloud_cover',
        'direction': 'asc'
    }]
)
items = search.item_collection()

# Load data
ds = stac.load(
    items,
    bands=['blue', 'green', 'red', 'nir', 'swir16'],
    resolution=10,
    chunks={},
    groupby='solar_day',
    preserve_original_order=True
)

# Select least cloudy scene
timestamp = pd.to_datetime(items[0].properties['datetime']).tz_convert(None)
scene = ds.sel(time=timestamp).compute()

# Apply scale and offset
scale = 0.0001
offset = -0.1
scene = scene.where(scene != 0) * scale + offset
```

## Visualize the Scene

```python
scene_da = scene.to_array('band')

# Create RGB preview
preview = scene_da.rio.reproject(scene_da.rio.crs, resolution=300)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
preview.sel(band=['red', 'green', 'blue']).plot.imshow(
    ax=ax, robust=True)
ax.set_title('True Color Composite')
ax.set_axis_off()
ax.set_aspect('equal')
plt.show()
```

## Calculate Spectral Indices

### NDVI

```python
# Calculate NDVI
ndvi = (scene.nir - scene.red) / (scene.nir + scene.red)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
ndvi.plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1, cbar_kwargs={'label': 'NDVI'})
ax.set_title('NDVI - Normalized Difference Vegetation Index')
ax.set_axis_off()
plt.show()
```

### NDWI

```python
# Calculate NDWI
ndwi = (scene.green - scene.nir) / (scene.green + scene.nir)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
ndwi.plot(ax=ax, cmap='Blues', vmin=-1, vmax=1, cbar_kwargs={'label': 'NDWI'})
ax.set_title('NDWI - Normalized Difference Water Index')
ax.set_axis_off()
plt.show()
```

### MNDWI

```python
# Calculate MNDWI
mndwi = (scene.green - scene.swir16) / (scene.green + scene.swir16)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
mndwi.plot(ax=ax, cmap='Blues', vmin=-1, vmax=1, cbar_kwargs={'label': 'MNDWI'})
ax.set_title('MNDWI - Modified Normalized Difference Water Index')
ax.set_axis_off()
plt.show()
```

### SAVI

```python
# Calculate SAVI
L = 0.5  # Soil brightness correction factor
savi = ((scene.nir - scene.red) * (1 + L)) / (scene.nir + scene.red + L)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
savi.plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1, cbar_kwargs={'label': 'SAVI'})
ax.set_title('SAVI - Soil Adjusted Vegetation Index')
ax.set_axis_off()
plt.show()
```

### EVI

```python
# Calculate EVI
evi = 2.5 * ((scene.nir - scene.red) / 
             (scene.nir + 6 * scene.red - 7.5 * scene.blue + 1))

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
evi.plot(ax=ax, cmap='RdYlGn', vmin=-1, vmax=1, cbar_kwargs={'label': 'EVI'})
ax.set_title('EVI - Enhanced Vegetation Index')
ax.set_axis_off()
plt.show()
```

### NDBI

```python
# Calculate NDBI
ndbi = (scene.swir16 - scene.nir) / (scene.swir16 + scene.nir)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
ndbi.plot(ax=ax, cmap='YlOrRd', vmin=-1, vmax=1, cbar_kwargs={'label': 'NDBI'})
ax.set_title('NDBI - Normalized Difference Built-up Index')
ax.set_axis_off()
plt.show()
```

## Multi-Index Comparison

```python
# Create subplot for multiple indices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# NDVI
ndvi.plot(ax=axes[0, 0], cmap='RdYlGn', vmin=-1, vmax=1, add_colorbar=False)
axes[0, 0].set_title('NDVI (Vegetation)')
axes[0, 0].set_axis_off()

# NDWI
ndwi.plot(ax=axes[0, 1], cmap='Blues', vmin=-1, vmax=1, add_colorbar=False)
axes[0, 1].set_title('NDWI (Water)')
axes[0, 1].set_axis_off()

# MNDWI
mndwi.plot(ax=axes[0, 2], cmap='Blues', vmin=-1, vmax=1, add_colorbar=False)
axes[0, 2].set_title('MNDWI (Water - Modified)')
axes[0, 2].set_axis_off()

# SAVI
savi.plot(ax=axes[1, 0], cmap='RdYlGn', vmin=-1, vmax=1, add_colorbar=False)
axes[1, 0].set_title('SAVI (Vegetation - Soil Adjusted)')
axes[1, 0].set_axis_off()

# EVI
evi.plot(ax=axes[1, 1], cmap='RdYlGn', vmin=-1, vmax=1, add_colorbar=False)
axes[1, 1].set_title('EVI (Enhanced Vegetation)')
axes[1, 1].set_axis_off()

# NDBI
ndbi.plot(ax=axes[1, 2], cmap='YlOrRd', vmin=-1, vmax=1, add_colorbar=False)
axes[1, 2].set_title('NDBI (Built-up)')
axes[1, 2].set_axis_off()

plt.tight_layout()
plt.show()
```

## Save Computed Indices

### Save as NetCDF

```python
# Combine indices into a dataset
indices = xr.Dataset({
    'ndvi': ndvi,
    'ndwi': ndwi,
    'mndwi': mndwi,
    'savi': savi,
    'evi': evi,
    'ndbi': ndbi
})

# Add metadata
indices.attrs['title'] = 'Spectral Indices'
indices.attrs['source'] = 'Sentinel-2'
indices.attrs['date'] = str(timestamp)

# Save to NetCDF
output_file = 'spectral_indices.nc'
indices.to_netcdf(output_file)
```

### Save as GeoTIFF

```python
# Save individual indices
ndvi.rio.to_raster('ndvi.tif')
ndwi.rio.to_raster('ndwi.tif')
mndwi.rio.to_raster('mndwi.tif')

# Save multi-band GeoTIFF
indices_da = indices.to_array('index')
indices_da.rio.to_raster('all_indices.tif')
```

### Save to Zarr

```python
# Save to Zarr (cloud-optimized)
indices.to_zarr('spectral_indices.zarr', mode='w', consolidated=True)
```

## Custom Index Functions

Create reusable functions for indices:

```python
def calculate_ndvi(nir, red):
    """Calculate NDVI."""
    return (nir - red) / (nir + red)

def calculate_ndwi(green, nir):
    """Calculate NDWI."""
    return (green - nir) / (green + nir)

def calculate_savi(nir, red, L=0.5):
    """Calculate SAVI."""
    return ((nir - red) * (1 + L)) / (nir + red + L)

def calculate_evi(nir, red, blue, G=2.5, C1=6, C2=7.5, L=1):
    """Calculate EVI."""
    return G * ((nir - red) / (nir + C1 * red - C2 * blue + L))

# Use functions
ndvi = calculate_ndvi(scene.nir, scene.red)
ndwi = calculate_ndwi(scene.green, scene.nir)
savi = calculate_savi(scene.nir, scene.red)
evi = calculate_evi(scene.nir, scene.red, scene.blue)
```

## Time Series of Indices

```python
# Load full time series
ds_full = stac.load(
    items,
    bands=['red', 'nir'],
    resolution=20,  # Coarser for faster processing
    chunks={'time': 10}
).compute()

# Apply scale/offset
ds_full = ds_full.where(ds_full != 0) * scale + offset

# Calculate NDVI time series
ndvi_ts = (ds_full.nir - ds_full.red) / (ds_full.nir + ds_full.red)

# Calculate spatial mean
ndvi_mean = ndvi_ts.mean(dim=['x', 'y'])

# Plot time series
fig, ax = plt.subplots(figsize=(12, 6))
ndvi_mean.plot(ax=ax, marker='o')
ax.set_title('NDVI Time Series')
ax.set_ylabel('Mean NDVI')
ax.grid(True, alpha=0.3)
plt.show()
```

## Thresholding and Classification

```python
# Classify NDVI into categories
ndvi_classified = xr.where(ndvi < 0, 0,  # Water/Non-vegetation
                  xr.where(ndvi < 0.2, 1,  # Bare soil
                  xr.where(ndvi < 0.5, 2,  # Sparse vegetation
                  xr.where(ndvi < 0.8, 3,  # Moderate vegetation
                           4))))  # Dense vegetation

# Define colors for each class
colors = ['blue', 'brown', 'yellow', 'lightgreen', 'darkgreen']
labels = ['Water', 'Bare Soil', 'Sparse Veg', 'Moderate Veg', 'Dense Veg']

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(ndvi_classified, cmap=plt.cm.colors.ListedColormap(colors))
ax.set_title('NDVI Classification')
ax.set_axis_off()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) 
                   for i in range(len(colors))]
ax.legend(handles=legend_elements, loc='upper right')
plt.show()
```

## Exercise

Calculate the Normalized Difference Snow Index (NDSI) using the formula:

$$NDSI = \frac{Green - SWIR}{Green + SWIR}$$

Then create a binary snow mask where NDSI > 0.4 indicates snow.

**Solution:**

```python
# Calculate NDSI
ndsi = (scene.green - scene.swir16) / (scene.green + scene.swir16)

# Create snow mask
snow_mask = ndsi > 0.4

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# NDSI
ndsi.plot(ax=axes[0], cmap='Blues', vmin=-1, vmax=1)
axes[0].set_title('NDSI - Normalized Difference Snow Index')
axes[0].set_axis_off()

# Snow mask
snow_mask.plot(ax=axes[1], cmap='Blues', add_colorbar=False)
axes[1].set_title('Snow Mask (NDSI > 0.4)')
axes[1].set_axis_off()

plt.tight_layout()
plt.show()
```

## Key Takeaways

!!! success "What You Learned"
    - Spectral indices combine bands to highlight specific features
    - NDVI measures vegetation health and density
    - NDWI and MNDWI detect water bodies
    - SAVI adjusts for soil brightness
    - EVI provides enhanced sensitivity in dense vegetation
    - NDBI identifies built-up areas
    - XArray makes index calculation straightforward
    - Indices can be saved in multiple formats
    - Time series analysis reveals temporal patterns
    - Thresholding enables simple classification

## Next Steps

→ Continue to [Cloud Masking](cloud-masking.md)

## Additional Resources

- [Index Database](https://www.indexdatabase.de/)
- [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices)
- [USGS Spectral Indices](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices)
- [Sentinel-2 Indices](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/)
