# Geemap Tiled Download

## Overview

**geemap** provides powerful functionality to download large-scale Earth Engine data directly to your local machine without using Earth Engine Compute Units (EECU) for exports. This is achieved through tiled downloading, where the area is split into manageable tiles, processed server-side, and downloaded directly.

**Key Benefits:**

- ✅ No EECU consumption for exports
- ✅ Direct download to local folder
- ✅ Automatic tiling for large areas
- ✅ Automatic merging of tiles
- ✅ Progress tracking
- ✅ Handles memory limitations

## Installation

```python
%%capture
!pip install geemap earthengine-api
```

## Basic Setup

```python
import ee
import geemap
import os

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='spatialgeography')

# Create output directory
output_folder = 'downloads'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
```

## Simple Tiled Download

### Download Sentinel-2 Composite

```python
# Define area of interest
roi = ee.Geometry.Rectangle([82.5, 27.0, 83.0, 27.5])

# Create Sentinel-2 composite
s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median() \
    .clip(roi)

# Select bands
image = s2.select(['B4', 'B3', 'B2', 'B8'])

# Download with automatic tiling
output_file = os.path.join(output_folder, 'sentinel2_composite.tif')

geemap.download_ee_image(
    image,
    filename=output_file,
    region=roi,
    scale=10,
    crs='EPSG:4326'
)

print(f"Downloaded: {output_file}")
```

## Advanced Tiled Download

### Large Area Download with Custom Tiling

```python
# Define large area (e.g., entire district)
large_roi = ee.Geometry.Rectangle([80.0, 25.0, 85.0, 30.0])

# Create image
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(large_roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .median() \
    .clip(large_roi)

# Select and scale bands
image = landsat.select(['SR_B4', 'SR_B3', 'SR_B2']) \
    .multiply(0.0000275).add(-0.2)

# Download with custom tile size
output_file = os.path.join(output_folder, 'landsat_large_area.tif')

geemap.download_ee_image(
    image,
    filename=output_file,
    region=large_roi,
    scale=30,
    crs='EPSG:4326',
    num_threads=4,  # Parallel downloads
    max_tile_size=1.0e8,  # 100 MB per tile
    max_tile_dim=10000  # Max 10000 pixels per dimension
)
```

## Download with Processing

### Calculate and Download NDVI

```python
# Define ROI
roi = ee.Geometry.Rectangle([82.5, 27.0, 83.5, 28.0])

# Get Sentinel-2 data
s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-06-01', '2023-08-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Cloud masking function
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Apply cloud mask and calculate NDVI
s2_masked = s2.map(mask_s2_clouds)
median = s2_masked.median()

# Calculate NDVI (processing done server-side)
ndvi = median.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Download NDVI
output_file = os.path.join(output_folder, 'ndvi_summer_2023.tif')

geemap.download_ee_image(
    ndvi,
    filename=output_file,
    region=roi,
    scale=10,
    crs='EPSG:4326'
)

print(f"NDVI downloaded: {output_file}")
```

## Multi-band Download

### Download Multiple Indices

```python
# Define ROI
roi = ee.Geometry.Rectangle([82.0, 26.5, 83.0, 27.5])

# Get Sentinel-2 median composite
s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()

# Calculate multiple indices (all processing server-side)
ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')

# Combine into multi-band image
indices = ndvi.addBands(ndwi).addBands(ndbi).clip(roi)

# Download all indices as multi-band GeoTIFF
output_file = os.path.join(output_folder, 'spectral_indices.tif')

geemap.download_ee_image(
    indices,
    filename=output_file,
    region=roi,
    scale=20,
    crs='EPSG:4326'
)

print(f"Multi-band indices downloaded: {output_file}")
```

## Time Series Download

### Download Monthly Composites

```python
import calendar

# Define ROI
roi = ee.Geometry.Rectangle([82.5, 27.0, 83.0, 27.5])

# Download monthly NDVI for 2023
for month in range(1, 13):
    # Get month start and end
    start_date = f'2023-{month:02d}-01'
    if month == 12:
        end_date = '2024-01-01'
    else:
        end_date = f'2023-{month+1:02d}-01'
    
    # Get Sentinel-2 data
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .median()
    
    # Calculate NDVI
    ndvi = s2.normalizedDifference(['B8', 'B4']).clip(roi)
    
    # Download
    month_name = calendar.month_name[month]
    output_file = os.path.join(output_folder, f'ndvi_{month:02d}_{month_name}.tif')
    
    try:
        geemap.download_ee_image(
            ndvi,
            filename=output_file,
            region=roi,
            scale=20,
            crs='EPSG:4326'
        )
        print(f"✓ Downloaded: {month_name}")
    except Exception as e:
        print(f"✗ Failed {month_name}: {str(e)}")
```

## Download from FeatureCollection

### Download Data for Administrative Boundaries

```python
# Load administrative boundaries
countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
india = countries.filter(ee.Filter.eq('country_na', 'India'))

# Get a specific state (example: Uttar Pradesh)
states = ee.FeatureCollection('FAO/GAUL/2015/level1')
state = states.filter(ee.Filter.eq('ADM1_NAME', 'Uttar Pradesh')).first()
state_geom = state.geometry()

# Get MODIS NDVI
modis = ee.ImageCollection('MODIS/006/MOD13A2') \
    .filterBounds(state_geom) \
    .filterDate('2023-01-01', '2023-12-31') \
    .select('NDVI') \
    .mean() \
    .clip(state_geom)

# Download
output_file = os.path.join(output_folder, 'uttar_pradesh_ndvi.tif')

geemap.download_ee_image(
    modis,
    filename=output_file,
    region=state_geom,
    scale=1000,
    crs='EPSG:4326'
)
```

## Advanced: Custom Tiling Strategy

### Manual Tile Control

```python
def download_large_area_custom(image, roi, output_file, tile_size=0.5):
    """
    Download large area with custom tiling.
    
    Parameters:
    -----------
    image : ee.Image
        Earth Engine image to download
    roi : ee.Geometry
        Region of interest
    output_file : str
        Output file path
    tile_size : float
        Tile size in degrees
    """
    import rasterio
    from rasterio.merge import merge
    import glob
    
    # Get bounds
    bounds = roi.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]
    
    # Create tiles
    temp_folder = os.path.join(output_folder, 'temp_tiles')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    tile_files = []
    tile_count = 0
    
    # Iterate over tiles
    for lon in range(int(min_lon), int(max_lon) + 1):
        for lat in range(int(min_lat), int(max_lat) + 1):
            # Create tile geometry
            tile_roi = ee.Geometry.Rectangle([
                lon, lat,
                min(lon + tile_size, max_lon),
                min(lat + tile_size, max_lat)
            ])
            
            # Download tile
            tile_file = os.path.join(temp_folder, f'tile_{tile_count}.tif')
            
            try:
                geemap.download_ee_image(
                    image,
                    filename=tile_file,
                    region=tile_roi,
                    scale=30,
                    crs='EPSG:4326'
                )
                tile_files.append(tile_file)
                tile_count += 1
                print(f"Downloaded tile {tile_count}")
            except Exception as e:
                print(f"Failed tile at ({lon}, {lat}): {str(e)}")
    
    # Merge tiles
    print("Merging tiles...")
    src_files_to_mosaic = []
    for tile_file in tile_files:
        src = rasterio.open(tile_file)
        src_files_to_mosaic.append(src)
    
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Save merged file
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Clean up temporary files
    for src in src_files_to_mosaic:
        src.close()
    
    for tile_file in tile_files:
        os.remove(tile_file)
    os.rmdir(temp_folder)
    
    print(f"✓ Merged file saved: {output_file}")

# Example usage
roi = ee.Geometry.Rectangle([80.0, 25.0, 82.0, 27.0])
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .median() \
    .clip(roi)

output_file = os.path.join(output_folder, 'large_area_merged.tif')
download_large_area_custom(landsat.select(['SR_B4', 'SR_B3', 'SR_B2']), 
                           roi, output_file, tile_size=0.5)
```

## Download with Visualization Parameters

### Apply Color Palette Before Download

```python
# Get elevation data
dem = ee.Image('USGS/SRTMGL1_003')
roi = ee.Geometry.Rectangle([82.0, 27.0, 83.0, 28.0])

# Clip to ROI
elevation = dem.clip(roi)

# Apply visualization (server-side)
vis_params = {
    'min': 0,
    'max': 3000,
    'palette': ['blue', 'green', 'yellow', 'orange', 'red']
}

# Convert to RGB
rgb = elevation.visualize(**vis_params)

# Download RGB image
output_file = os.path.join(output_folder, 'elevation_colored.tif')

geemap.download_ee_image(
    rgb,
    filename=output_file,
    region=roi,
    scale=30,
    crs='EPSG:4326'
)
```

## Batch Download

### Download Multiple Regions

```python
# Define multiple ROIs
regions = {
    'region_1': ee.Geometry.Rectangle([82.0, 27.0, 82.5, 27.5]),
    'region_2': ee.Geometry.Rectangle([82.5, 27.0, 83.0, 27.5]),
    'region_3': ee.Geometry.Rectangle([82.0, 26.5, 82.5, 27.0]),
}

# Get Sentinel-2 composite
s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()

# Download each region
for region_name, roi in regions.items():
    image = s2.clip(roi).select(['B4', 'B3', 'B2'])
    output_file = os.path.join(output_folder, f'{region_name}_s2.tif')
    
    try:
        geemap.download_ee_image(
            image,
            filename=output_file,
            region=roi,
            scale=10,
            crs='EPSG:4326'
        )
        print(f"✓ Downloaded: {region_name}")
    except Exception as e:
        print(f"✗ Failed {region_name}: {str(e)}")
```

## Progress Tracking

### Download with Progress Bar

```python
from tqdm import tqdm

def download_with_progress(image, roi, output_file, scale=10):
    """Download with progress tracking."""
    
    # Calculate approximate number of tiles
    bounds = roi.bounds().getInfo()['coordinates'][0]
    width = bounds[2][0] - bounds[0][0]
    height = bounds[2][1] - bounds[0][1]
    
    # Estimate tiles (rough approximation)
    tile_size_deg = 0.1  # ~10km
    n_tiles = int((width / tile_size_deg) * (height / tile_size_deg))
    
    print(f"Downloading {n_tiles} estimated tiles...")
    
    with tqdm(total=100, desc="Downloading") as pbar:
        geemap.download_ee_image(
            image,
            filename=output_file,
            region=roi,
            scale=scale,
            crs='EPSG:4326'
        )
        pbar.update(100)
    
    print(f"✓ Download complete: {output_file}")

# Example usage
roi = ee.Geometry.Rectangle([82.0, 27.0, 83.0, 28.0])
s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .median() \
    .clip(roi)

output_file = os.path.join(output_folder, 'sentinel2_with_progress.tif')
download_with_progress(s2.select(['B4', 'B3', 'B2']), roi, output_file)
```

## Best Practices

### 1. Choose Appropriate Scale

```python
# Sentinel-2: 10m for RGB/NIR, 20m for other bands
scale = 10

# Landsat: 30m
scale = 30

# MODIS: 250m, 500m, or 1000m depending on product
scale = 1000
```

### 2. Limit Area Size

```python
# For high resolution (10m), keep area < 100 km²
# For medium resolution (30m), keep area < 500 km²
# For coarse resolution (1000m), can handle larger areas
```

### 3. Use Appropriate CRS

```python
# The global standard for geospatial data is geographic
crs = 'EPSG:4326'
```

### 4. Handle Errors Gracefully

```python
try:
    geemap.download_ee_image(image, filename=output_file, ...)
except Exception as e:
    print(f"Download failed: {str(e)}")
    # Retry with smaller tiles or coarser resolution
```

## Key Takeaways

!!! success "What You Learned"
    - geemap enables direct downloads without EECU usage
    - Automatic tiling handles large areas efficiently
    - Processing happens server-side in Earth Engine
    - Tiles are automatically merged into final GeoTIFF
    - Supports multi-band and time series downloads
    - Custom tiling strategies for maximum control
    - Progress tracking for long downloads
    - No export tasks or Drive storage needed

## Next Steps

→ See [Advanced Topics](../advanced/scaling-dask.md) for more optimization techniques

## Additional Resources

- [geemap Documentation](https://geemap.org/)
- [geemap Download Examples](https://geemap.org/notebooks/71_timelapse/)
- [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)
- [geemap GitHub](https://github.com/gee-community/geemap)
