# Python-First Data Access & Local Processing

Break free from cloud-only sandboxes. Learn how to use Google Earth Engine purely as a data provider, while performing all your analysis, modeling, and visualization in a local, open-source Python environment.

## Overview

Total dependence on a single proprietary cloud platform (like GEE) carries significant risks, including vendor lock-in and potential loss of work. This workflow demonstrates a **"Cloud-Hybrid"** approach:

1. **GEE as a Data Source**: Use XEE to interactively explore and select data.
2. **Advanced Vector Data**: Use `OSMnx` to fetch and analyze complex OpenStreetMap networks locally.
3. **Tiled Download**: Use `geemap` to export large-scale raster data to your local machine.
4. **Local Analysis**: Perform all subsequent processing using `Xarray`, `Dask`, and `Rioxarray`.

## Step 1: Initialize and Preview with XEE

We'll use XEE for a quick preview. Note that we are only using it to *see* what we want before we download it for real work.

```python
import ee
import xarray as xr
import xee
import geemap
import os
import osmnx as ox

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define a Region of Interest (ROI)
roi_point = [77.1025, 28.7041]
roi = ee.Geometry.Point(roi_point).buffer(5000).bounds()

# Preview Sentinel-2 data
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-06-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .median()

# Quick interactive peek
ds = xr.open_dataset(s2, engine='ee', geometry=roi, scale=10)
print(ds)
```

## Step 2: Advanced OSM Data with OSMnx

While `geemap` is great for quick OSM fetches, `OSMnx` is the gold standard for Python-based street network analysis. It allows you to download and model street networks as `NetworkX` graphs.

```python
# Download the street network for a building-centric analysis
# We'll use the same coordinates as our ROI
location_point = (28.7041, 77.1025)
G = ox.graph_from_point(location_point, dist=2000, network_type='drive')

# Plot the network locally (no cloud needed!)
fig, ax = ox.plot_graph(G, node_size=0, edge_color='gray')

# Fetch specialized POIs (e.g., hospitals)
hospitals = ox.features_from_point(location_point, tags={'amenity': 'hospital'}, dist=2000)
print(f"Number of hospitals found: {len(hospitals)}")
```

## Step 3: Tiled Downloading (The Escape Hatch)

To ensure your work is portable and independent of Google's servers, download the raster data to your local disk.

```python
output_dir = 'local_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'study_area_s2.tif')

# Tiled Download: Bypasses GEE's 5,000-pixel export limit
geemap.download_ee_image(
    s2.select(['B4', 'B3', 'B2', 'B8']), # Red, Green, Blue, NIR
    filename=output_file,
    region=roi,
    scale=10,
    num_threads=8 # Fast parallel downloading
)
```

## Step 4: Full Local Analysis with Xarray & Dask

Now that the data is on your disk, you are free to use any Python library (SciPy, Scikit-Learn, PyTorch) for your analysis.

```python
import rioxarray

# Open the local file
# Dask enables out-of-core computation for files larger than RAM
local_ds = rioxarray.open_rasterio(output_file, chunks={'x': 512, 'y': 512})

# Calculate NDVI locally
ndvi = (local_ds.sel(band=4) - local_ds.sel(band=1)) / (local_ds.sel(band=4) + local_ds.sel(band=1))

# Save results locally
ndvi.rio.to_raster(os.path.join(output_dir, 'local_ndvi.tif'))

# Visualize
ndvi.plot(cmap='RdYlGn')
```

## Why This Matters

| Category | Proprietary (GEE Only) | Python-First (Cloud-Hybrid) |
| :--- | :--- | :--- |
| **Sustainability** | Dependent on Google's mercy | Fully portable and independent |
| **Innovation** | Limited to GEE-supported functions | Use any Python library (SciPy, AI) |
| **Patents** | Hard to claim "Author" rights | Direct claim on innovative code |
| **Performance** | Throttled by Google's servers | Limited only by your hardware/Dask |

## Key Takeaways

!!! success "Independence is Power"
    - Use **XEE** for discovery, but **geemap/OSMnx** for extraction.
    - Treat **Local Storage** as your primary workspace for proprietary research.
    - **Python** is the real engine; cloud platforms are just the fuel.

→ Next: [Indices and Enhancement](indices-enhancement.md)
