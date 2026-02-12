# NDVI Analysis using XEE

This example demonstrates how to perform a complete NDVI analysis workflow using Earth Engine and XArray (via XEE). We'll cover data selection, cloud masking, spatial alignment, and visualization.

## Scenario

**Objective**: Calculate and visualize the median NDVI for a buffered region around Visakhapatnam, India for the year 2020.
**Dataset**: Sentinel-2 Harmonized Surface Reflectance.
**Area**: 10km buffer around Vizag.

## Implementation

### 1. Setup and Initialization

First, we'll import the necessary libraries and initialize Earth Engine.

```python
import xarray as xr
import xee
import ee
import geemap
import matplotlib.pyplot as plt

# Initialize Earth Engine with project ID
ee.Initialize(project='spatialgeography')
```

### 2. Define Region and Dataset

We'll define a point of interest and create a 5km buffer (10km total width) as our Region of Interest (ROI). Then, we'll filter the Sentinel-2 collection.

```python
def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask)

# Define center point (Vizag) and buffer area
center_point = [83.277, 17.7009]
roi = ee.Geometry.Point(center_point).buffer(5000).bounds()

# Filter collection
dataset = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2020-01-01', '2020-12-31')
    # Relaxed cloud threshold to ensure data coverage
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(mask_s2_clouds)
)

print(f"Collection size: {dataset.size().getInfo()} images")
```

### 3. Interactive Visualization

Before deep processing, it's always good to see the data using `geemap`.

```python
visualization = {
    'min': 0, 
    'max': 3000, 
    'bands': ['B4', 'B3', 'B2']
}

m = geemap.Map()
m.set_center(center_point[0], center_point[1], 12)
m.add_layer(dataset.median(), visualization, 'RGB')
m
```

### 4. XEE Processing and Plotting

Now, we'll open the dataset using the `ee` engine in XArray, calculate NDVI, and plot the result.

```python
print("Opening dataset with XEE...")

# Open the dataset with explicit projection
ds_xee = xr.open_dataset(
    dataset,
    engine='ee',
    geometry=roi,
    scale=100,  # 100m for consistent analysis
    crs='EPSG:4326', # WGS 84
    ee_mask_value=-9999
)

# Calculate NDVI (B8 is NIR, B4 is Red)
ndvi_xee = (ds_xee.B8 - ds_xee.B4) / (ds_xee.B8 + ds_xee.B4)

# Calculate temporal median (always sort by time first!)
ndvi_median = ndvi_xee.sortby('time').median(dim='time')

# Trigger computation
ndvi_result = ndvi_median.compute()

# Check for valid data and plot
if ndvi_result.notnull().any():
    print("Plotting NDVI map...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ndvi_result.plot(ax=ax, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Median NDVI (Vizag 2020)\nScale: 100m, CRS: EPSG:4326')
    plt.show()
else:
    print("WARNING: All pixels are NaN! Check cloud filters.")
```

## Key Best Practices

!!! tip "Relax Cloud Filters"
    In tropical or coastal regions (like Vizag), strict cloud filters (e.g., < 20%) can result in empty datasets. Relaxing the filter to 80% and using a median composite often yields better results.

!!! tip "CRS Selection"
    While `EPSG:3857` (Web Mercator) is great for visualization, `EPSG:4326` is the global standard for geospatial analysis. Choose based on your requirements.

!!! tip "Explicit .compute()"
    Remember that XEE datasets are lazy. You must call `.compute()` on your final array to trigger the Earth Engine computation and download the results into memory.
