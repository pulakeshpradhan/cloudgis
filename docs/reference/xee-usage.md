# Google Earth Engine Reference

## Overview

Google Earth Engine (GEE) is a cloud-based platform for planetary-scale geospatial analysis. It provides access to petabytes of satellite imagery and geospatial datasets with powerful processing capabilities.

## Key Features

- **Massive Data Catalog**: 1000+ public datasets
- **Server-side Processing**: Computation happens in Google's cloud
- **Parallel Processing**: Automatic parallelization
- **No Download Required**: Stream data directly
- **Free for Research**: No cost for non-commercial use

## Authentication

### First-Time Setup

```python
import ee

# Authenticate (opens browser)
ee.Authenticate()

# Initialize
ee.Initialize(project='spatialgeography')
```

### Project-based Authentication

```python
# For newer accounts
ee.Initialize(project='spatialgeography')
```

### Service Account

```python
# For production/automated systems
credentials = ee.ServiceAccountCredentials(
    email='your-service-account@project.iam.gserviceaccount.com',
    key_file='/path/to/private-key.json'
)
ee.Initialize(credentials)
```

## Core Data Types

### ee.Image

A single raster image with one or more bands.

```python
# Load image
image = ee.Image('COPERNICUS/S2_SR/20230115T051131_20230115T051126_T44QMG')

# Select bands
rgb = image.select(['B4', 'B3', 'B2'])

# Get info
print(image.bandNames().getInfo())
print(image.projection().getInfo())
```

### ee.ImageCollection

A stack or time series of images.

```python
# Load collection
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(ee.Geometry.Point([82.5, 27.0])) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Get size
print(collection.size().getInfo())

# Get first image
first = collection.first()
```

### ee.Geometry

Geometric objects (points, lines, polygons).

```python
# Point
point = ee.Geometry.Point([82.5, 27.0])

# Rectangle
rect = ee.Geometry.Rectangle([82.0, 26.5, 83.0, 27.5])

# Polygon
polygon = ee.Geometry.Polygon([[
    [82.0, 27.0],
    [83.0, 27.0],
    [83.0, 28.0],
    [82.0, 28.0],
    [82.0, 27.0]
]])

# Buffer
buffered = point.buffer(10000)  # 10km buffer
```

### ee.Feature

A geometry with properties (attributes).

```python
# Create feature
feature = ee.Feature(
    ee.Geometry.Point([82.5, 27.0]),
    {'name': 'Location A', 'value': 100}
)

# Get property
name = feature.get('name')
```

### ee.FeatureCollection

A collection of features (vector data).

```python
# Load feature collection
countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')

# Filter
india = countries.filter(ee.Filter.eq('country_na', 'India'))

# Get geometry
geometry = india.geometry()
```

## Filtering

### Spatial Filtering

```python
# Filter by bounds
filtered = collection.filterBounds(geometry)

# Filter by geometry intersection
filtered = collection.filter(ee.Filter.intersects('.geo', geometry))
```

### Temporal Filtering

```python
# Date range
filtered = collection.filterDate('2023-01-01', '2023-12-31')

# Specific dates
filtered = collection.filter(
    ee.Filter.inList('system:index', ['20230115', '20230120'])
)
```

### Metadata Filtering

```python
# Less than
filtered = collection.filter(
    ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)
)

# Greater than
filtered = collection.filter(
    ee.Filter.gt('SUN_ELEVATION', 30)
)

# Equals
filtered = collection.filter(
    ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2A')
)

# In list
filtered = collection.filter(
    ee.Filter.inList('MGRS_TILE', ['44QMG', '44QNG'])
)

# Multiple conditions (AND)
filtered = collection.filter(
    ee.Filter.And(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20),
        ee.Filter.gt('SUN_ELEVATION', 30)
    )
)

# Multiple conditions (OR)
filtered = collection.filter(
    ee.Filter.Or(
        ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2A'),
        ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2B')
    )
)
```

## Mapping Functions

### map()

Apply function to each image in collection.

```python
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

collection_with_ndvi = collection.map(add_ndvi)
```

### Cloud Masking

```python
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

masked = collection.map(mask_s2_clouds)
```

## Reducers

### Temporal Reduction

```python
# Median composite
median = collection.median()

# Mean
mean = collection.mean()

# Max
maximum = collection.max()

# Percentile
p90 = collection.reduce(ee.Reducer.percentile([90]))
```

### Spatial Reduction

```python
# Mean over region
mean_value = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geometry,
    scale=30,
    maxPixels=1e9
)

# Multiple statistics
stats = image.reduceRegion(
    reducer=ee.Reducer.mean() \
        .combine(ee.Reducer.stdDev(), '', True) \
        .combine(ee.Reducer.min(), '', True) \
        .combine(ee.Reducer.max(), '', True),
    geometry=geometry,
    scale=30
)
```

### Zonal Statistics

```python
# Statistics by zone
zonal_stats = image.reduceRegions(
    collection=zones,
    reducer=ee.Reducer.mean(),
    scale=30
)
```

## Image Operations

### Band Math

```python
# NDVI
ndvi = image.normalizedDifference(['B8', 'B4'])

# Custom calculation
evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }
)

# Arithmetic
result = image.select('B8').subtract(image.select('B4'))
result = image.select('B8').divide(image.select('B4'))
```

### Masking

```python
# Mask by value
masked = image.updateMask(image.select('NDVI').gt(0.3))

# Mask by another image
mask = image.select('QA').eq(0)
masked = image.updateMask(mask)
```

### Clipping

```python
# Clip to geometry
clipped = image.clip(geometry)

# Clip collection
clipped_collection = collection.map(lambda img: img.clip(geometry))
```

## Exporting

### Export to Drive

```python
# Export image
task = ee.batch.Export.image.toDrive(
    image=image,
    description='my_export',
    folder='EarthEngine',
    fileNamePrefix='sentinel2_image',
    scale=10,
    region=geometry,
    maxPixels=1e9
)
task.start()

# Check status
print(task.status())
```

### Export to Asset

```python
task = ee.batch.Export.image.toAsset(
    image=image,
    description='export_to_asset',
    assetId='users/username/my_image',
    scale=10,
    region=geometry
)
task.start()
```

### Export to Cloud Storage

```python
task = ee.batch.Export.image.toCloudStorage(
    image=image,
    description='export_to_gcs',
    bucket='my-bucket',
    fileNamePrefix='sentinel2',
    scale=10,
    region=geometry
)
task.start()
```

### Export Table

```python
task = ee.batch.Export.table.toDrive(
    collection=feature_collection,
    description='export_table',
    fileFormat='CSV'
)
task.start()
```

## Common Datasets

### Sentinel-2

```python
# Surface Reflectance
s2_sr = ee.ImageCollection('COPERNICUS/S2_SR')

# Top of Atmosphere
s2_toa = ee.ImageCollection('COPERNICUS/S2')

# Harmonized (2015-present)
s2_harmonized = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
```

### Landsat

```python
# Landsat 8 Collection 2 Level 2
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

# Landsat 9
l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
```

### MODIS

```python
# NDVI
modis_ndvi = ee.ImageCollection('MODIS/006/MOD13A2')

# Land Surface Temperature
modis_lst = ee.ImageCollection('MODIS/006/MOD11A2')

# Land Cover
modis_lc = ee.ImageCollection('MODIS/006/MCD12Q1')
```

### Climate Data

```python
# ERA5 Daily
era5 = ee.ImageCollection('ECMWF/ERA5/DAILY')

# CHIRPS Precipitation
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

# TerraClimate
terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
```

### Terrain

```python
# SRTM DEM
srtm = ee.Image('USGS/SRTMGL1_003')

# ALOS World 3D
alos = ee.Image('JAXA/ALOS/AW3D30/V3_2')
```

## Best Practices

### 1. Filter Early

```python
# Good: Filter before processing
filtered = collection \
    .filterBounds(geometry) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
result = filtered.median()

# Bad: Process then filter
result = collection.median()
filtered = result.clip(geometry)
```

### 2. Use Appropriate Scale

```python
# Good: Match data resolution
stats = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geometry,
    scale=10  # Sentinel-2 resolution
)

# Bad: Too fine (slow, unnecessary)
stats = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geometry,
    scale=1
)
```

### 3. Limit Computation

```python
# Good: Use maxPixels
stats = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geometry,
    scale=30,
    maxPixels=1e9
)

# Good: Use smaller geometry
small_geometry = geometry.buffer(-1000)
```

### 4. Use getInfo() Sparingly

```python
# Good: Minimize getInfo() calls
size = collection.size().getInfo()
print(f"Collection has {size} images")

# Bad: getInfo() in loop
for i in range(collection.size().getInfo()):
    image = ee.Image(collection.toList(1, i).get(0))
    print(image.get('system:index').getInfo())
```

## Debugging

### Print Information

```python
# Print to console
print(image.bandNames().getInfo())
print(collection.size().getInfo())
print(geometry.area().getInfo())
```

### Inspect Properties

```python
# Get all properties
props = image.propertyNames().getInfo()
print(props)

# Get specific property
cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
```

### Visualize

```python
# In Jupyter with geemap
import geemap

Map = geemap.Map()
Map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000})
Map.centerObject(geometry, 10)
Map
```

## Computational Limits

### Free Tier Limits

- **Concurrent requests**: 3,000
- **User memory**: 256 MB per request
- **Computation time**: 5 minutes per request
- **Export size**: 10 GB per file
- **Asset storage**: 250 GB

### Optimization Tips

1. **Use smaller geometries**
2. **Reduce spatial resolution** (increase scale)
3. **Filter collections** before processing
4. **Use appropriate reducers**
5. **Batch exports** for large areas

## Common Issues & Fixes (XEE-specific)

### 1. The Monotonicity Error

**Error**: `ValueError: Index must be monotonic for resampling`
**Cause**: Earth Engine collections are not always sorted by time when streamed through XEE.
**Solution**: Use `.sortby('time')` before any resampling or interpolation.

```python
ds = xr.open_dataset(collection, engine='ee')
ds = ds.sortby('time')  # Fix
resampled = ds.resample(time='1M').median()
```

### 2. All-NaN Data (Empty Plot)

**Error**: `TypeError: No numeric data to plot.`
**Cause**: Aggressive cloud masking or strict cloud filters (e.g., `<20%`) removed all valid pixels.
**Solution**: Increase the `CLOUDY_PIXEL_PERCENTAGE` filter or check `ds.notnull().any()` before plotting.

### 3. Spatial Dimension Mismatch

**Error**: `KeyError: 'X'` or `KeyError: 'Y'`
**Cause**: XEE dimension names can vary (`lon/lat` vs `X/Y`) based on the CRS used.
**Solution**: Dynamically detect spatial dimensions or explicitly specify **EPSG:4326** for consistency.

```python
spatial_dims = [d for d in ds.dims if d not in ['time', 'band']]
ds.mean(dim=spatial_dims).plot()
```

## Additional Resources

... (previous content) ...

## Quick Reference

### Common Band Names

**Sentinel-2:**

- B1: Coastal aerosol (443nm)
- B2: Blue (490nm)
- B3: Green (560nm)
- B4: Red (665nm)
- B8: NIR (842nm)
- B11: SWIR1 (1610nm)
- B12: SWIR2 (2190nm)

**Landsat 8/9:**

- B1: Coastal/Aerosol
- B2: Blue
- B3: Green
- B4: Red
- B5: NIR
- B6: SWIR1
- B7: SWIR2

### Common Filters

```python
# Date
.filterDate('2023-01-01', '2023-12-31')

# Bounds
.filterBounds(geometry)

# Cloud cover
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Property equals
.filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2A'))
```

### Common Reducers

```python
ee.Reducer.mean()
ee.Reducer.median()
ee.Reducer.min()
ee.Reducer.max()
ee.Reducer.stdDev()
ee.Reducer.sum()
ee.Reducer.count()
ee.Reducer.percentile([10, 50, 90])
```
