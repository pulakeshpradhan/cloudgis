# Data Access and Tiled Downloading

Explore Earth Engine data interactively with XEE and download large datasets (Satellite & OSM) locally using geemap and OSMnx.

## Overview

This example demonstrates the complete workflow for:

1. **Direct Data Access**: Load Earth Engine data into an XArray dataset using XEE.
2. **OSM Data Retrieval**: Get OpenStreetMap vector data (buildings, roads, etc.) using geemap.
3. **Tiled Downloading**: Use `geemap` to download high-resolution raster data in chunks.

## Step 1: Initialize and Load Data with XEE

We'll start by loading Sentinel-2 data for a region of interest (ROI) to perform some quick analysis.

```python
import ee
import xarray as xr
import xee
import geemap
import os

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define a Region of Interest (ROI)
roi = ee.Geometry.Point([77.1025, 28.7041]).buffer(5000).bounds()

# Load Sentinel-2 SR data
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-06-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .median() \
    .clip(roi)

# Load with XEE for interactive analysis
# This allows you to work with the data as an XArray object
ds = xr.open_dataset(s2, engine='ee', geometry=roi, scale=10)
print(ds)
```

## Step 2: OpenStreetMap (OSM) Data Access

`geemap` provides a convenient way to fetch OSM vector data directly into your notebook.

```python
# Get buildings in the area
buildings = geemap.osm_to_gdf(roi, tags={'building': True})

# Get roads/highways
roads = geemap.osm_to_gdf(roi, tags={'highway': True})

print(f"Number of buildings found: {len(buildings)}")
print(f"Number of road segments: {len(roads)}")

# Visualize on a Map
Map = geemap.Map()
Map.centerObject(roi, 14)
Map.addLayer(s2, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'Satellite')
Map.add_gdf(buildings, layer_name='Buildings', fill_color='red')
Map.add_gdf(roads, layer_name='Roads', color='blue')
Map
```

## Step 3: Tiled Downloading with geemap

When you need the actual GeoTIFF files on your disk for local processing or inclusion in a GIS, `geemap`'s `download_ee_image` is the most efficient way to get high-resolution data without using Earth Engine's standard export tasks.

```python
# Create an output directory
output_dir = 'downloads'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the output file path
output_file = os.path.join(output_dir, 'delhi_sentinel2_10m.tif')

# Tiled Download
# geemap automatically splits the large request into tiles, 
# downloads them, and merges them into a single GeoTIFF.
geemap.download_ee_image(
    s2.select(['B4', 'B3', 'B2', 'B8']), # Select essential bands
    filename=output_file,
    region=roi,
    scale=10,        # 10m resolution
    crs='EPSG:4326',  # WGS 84
    num_threads=4    # Uses parallel downloads for speed
)

print(f"Dataset downloaded successfully to: {output_file}")
```

## Step 4: Verify Downloaded Data Locally

Once downloaded, you can load the local file back into XArray using `rioxarray` to verify the content.

```python
import rioxarray

# Load the local GeoTIFF
local_ds = rioxarray.open_rasterio(output_file)
print(f"Local Dataset Shape: {local_ds.shape}")

# Plot a single band
local_ds.sel(band=1).plot(cmap='viridis')
```

## Why Use This Workflow?

| Feature | XEE Access | geemap Tiled Download |
| :--- | :--- | :--- |
| **Primary Use** | Real-time analysis, visualization | Local storage, GIS integration |
| **Compute** | Dynamic, on-the-fly | Pre-computed, downloaded |
| **Advantages** | No local storage needed | Full resolution, offline access |
| **Scale** | Best for smaller areas or coarse res | Handles large areas via tiling |

## Key Takeaways

!!! success "Summary"
    - **XEE** is your go-to for exploratory data analysis directly in your Python notebook.
    - **geemap** provides the bridge for getting high-quality data out of the cloud and onto your local machine.
    - **Tiling** bypasses the standard 5,000-pixel limit of the Earth Engine GetThumbURL/Download API.
    - **OSM Integration** allows you to seamlessly combine satellite imagery with vector ground-truth data.

→ Next: [Indices and Enhancement](indices-enhancement.md)
