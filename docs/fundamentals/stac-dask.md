# STAC and Dask Basics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spatialthoughts/courses/blob/master/code/python_remote_sensing/02_stac_dask_basics.ipynb)

## Overview

In this section, we'll learn the basics of querying cloud-hosted data via [STAC](https://stacspec.org/en) and leverage parallel computing via [Dask](https://tutorial.xarray.dev/intermediate/xarray_and_dask.html).

We will learn how to:

- Query a catalog of Sentinel-2 images
- Find the least-cloudy scene over a chosen area
- Visualize the scene
- Download it as a GeoTIFF file

## Setup and Data Download

Install required packages:

```python
%%capture
if 'google.colab' in str(get_ipython()):
    !pip install pystac-client odc-stac rioxarray dask jupyter-server-proxy
```

Import libraries:

```python
import os
import matplotlib.pyplot as plt
import pandas as pd
import pystac_client
from odc import stac
import xarray as xr
import rioxarray as rxr
```

## Dask

[Dask](https://www.dask.org/) is a Python library to run your computation in parallel across many machines. Dask has built-in support for key geospatial packages like XArray and Pandas, allowing you to scale your computation easily.

**Key Features:**

- Run code in parallel on your laptop, cloud machine, or cluster
- Seamless integration with XArray and Pandas
- Lazy evaluation for efficient computation
- Interactive dashboard for monitoring

### Starting a Dask Client

```python
from dask.distributed import Client

client = Client()  # set up local cluster on the machine
client
```

### Viewing Dask Dashboard in Colab

If running in Colab, create a proxy URL to view the dashboard:

```python
if 'google.colab' in str(get_ipython()):
    from google.colab import output
    port_to_expose = 8787  # Default port for Dask dashboard
    print(output.eval_js(f'google.colab.kernel.proxyPort({port_to_expose})'))
```

## Spatio Temporal Asset Catalog (STAC)

STAC is an open standard for specifying and querying geospatial data. Data providers can share catalogs of:

- Satellite imagery
- Climate datasets
- LIDAR data
- Vector data

All STAC catalogs can be queried to find matching assets by time, location, or metadata.

### STAC Components

**Item**: A single spatiotemporal asset (e.g., one satellite scene)

```json
{
  "type": "Feature",
  "stac_version": "1.0.0",
  "id": "S2A_MSIL2A_20230115",
  "properties": {
    "datetime": "2023-01-15T10:30:00Z",
    "eo:cloud_cover": 15.5
  },
  "geometry": {...},
  "assets": {
    "red": {"href": "s3://..."},
    "nir": {"href": "s3://..."}
  }
}
```

**Collection**: A group of related items

**Catalog**: A collection of collections

**API**: RESTful interface for searching

### Browse Available Catalogs

Visit [https://stacindex.org/](https://stacindex.org/) to explore available STAC catalogs.

### Connecting to a STAC Catalog

Let's use [Earth Search by Element 84](https://stacindex.org/catalogs/earth-search#/) to access Sentinel-2 data on AWS:

```python
catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1')
```

### Defining Search Parameters

```python
latitude = 27.163
longitude = 82.608
year = 2023

# Define bounding box around the point
km2deg = 1.0 / 111
x, y = (longitude, latitude)
r = 1 * km2deg  # radius in degrees
bbox = (x - r, y - r, x + r, y + r)
```

### Basic Search

```python
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime=f'{year}'
)
items = search.item_collection()
items
```

### Filtering by Metadata

Apply additional filters for cloud cover and nodata pixels:

```python
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime=f'{year}',
    query={
        'eo:cloud_cover': {'lt': 30},
        's2:nodata_pixel_percentage': {'lt': 10}
    }
)
items = search.item_collection()
```

### Sorting Results

Sort by cloud cover to get the clearest scenes first:

```python
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
items
```

## Load STAC Images to XArray

Load matching images as an XArray Dataset:

```python
ds = stac.load(
    items,
    bands=['red', 'green', 'blue', 'nir'],
    resolution=10,
    chunks={},  # <-- use Dask
    groupby='solar_day',
    preserve_original_order=True
)
ds
```

### Check Dataset Size

```python
print(f'DataSet size: {ds.nbytes/1e6:.2f} MB.')
```

## Select a Single Scene

Get the timestamp of the least cloudy scene:

```python
timestamp = pd.to_datetime(items[0].properties['datetime']).tz_convert(None)
scene = ds.sel(time=timestamp)
scene
```

Check scene size:

```python
print(f'Scene size: {scene.nbytes/1e6:.2f} MB.')
```

### Load Data into Memory

Use Dask to parallelize data loading:

```python
%%time
scene = scene.compute()
```

Watch the Dask dashboard to see the parallel processing in action!

### Handle NoData Values

Sentinel-2 scenes have NoData value of 0:

```python
scene = scene.where(scene != 0)
scene
```

### Apply Scale and Offset

Convert raw pixel values to reflectances:

```python
scale = 0.0001
offset = -0.1
scene = scene * scale + offset
```

!!! info "Scale and Offset Values"
    For Sentinel-2 scenes captured after Jan 25, 2022:

    ```text
    Scale: 0.0001
    Offset: -0.1
    ```

    These values are in the `raster:bands` metadata for each band.

## Visualize the Scene

Convert Dataset to DataArray:

```python
scene_da = scene.to_array('band')
scene_da
```

### Check Spatial Metadata

```python
print('CRS:', scene_da.rio.crs)
print('Resolution:', scene_da.rio.resolution())
```

### Create Preview

Resample to lower resolution for visualization:

```python
preview = scene_da.rio.reproject(
    scene_da.rio.crs, resolution=300
)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 5)
preview.sel(band=['red', 'green', 'blue']).plot.imshow(
    ax=ax,
    robust=True)
ax.set_title('RGB Visualization')
ax.set_axis_off()
ax.set_aspect('equal')
plt.show()
```

### False Color Composite

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
preview.sel(band=['nir', 'red', 'green']).plot.imshow(
    ax=ax,
    robust=True)
ax.set_title('False Color (NIR-R-G)')
ax.set_axis_off()
ax.set_aspect('equal')
plt.show()
```

## Save Results

### Save as NetCDF

```python
output_file = 'scene.nc'
scene.to_netcdf(output_file)
```

### Save as GeoTIFF

```python
# Save RGB composite
rgb_file = 'rgb_composite.tif'
scene_da.sel(band=['red', 'green', 'blue']).rio.to_raster(rgb_file)

# Save single band
nir_file = 'nir_band.tif'
scene.nir.rio.to_raster(nir_file)
```

### Save to Google Drive

```python
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
    
    drive_path = '/content/drive/MyDrive/remote-sensing-outputs'
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)
    
    output_path = os.path.join(drive_path, 'scene.nc')
    scene.to_netcdf(output_path)
```

## Advanced STAC Queries

### Search by Geometry

```python
from shapely.geometry import Point

# Create a point geometry
point = Point(longitude, latitude)
buffer = point.buffer(0.01)  # ~1km buffer

search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    intersects=buffer.__geo_interface__,
    datetime='2023-01-01/2023-12-31'
)
```

### Multiple Collections

```python
search = catalog.search(
    collections=['sentinel-2-c1-l2a', 'landsat-c2-l2'],
    bbox=bbox,
    datetime=f'{year}'
)
```

### Complex Queries

```python
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime='2023-06-01/2023-08-31',  # Summer months
    query={
        'eo:cloud_cover': {'lt': 20},
        's2:nodata_pixel_percentage': {'lt': 5},
        'platform': {'in': ['sentinel-2a', 'sentinel-2b']}
    },
    limit=50
)
```

## Dask Best Practices

### Chunk Size Selection

```python
# Good chunking - balanced chunks
ds = stac.load(
    items,
    bands=['red', 'nir'],
    chunks={'time': 10, 'x': 512, 'y': 512}
)

# Too small - overhead dominates
ds_bad = stac.load(items, chunks={'time': 1, 'x': 64, 'y': 64})

# Too large - memory issues
ds_bad = stac.load(items, chunks={'time': 100, 'x': 4096, 'y': 4096})
```

### Monitor Performance

```python
# View task graph
ds.red.data.visualize(filename='task_graph.png')

# Check chunk info
print(ds.red.data)
```

### Persist Results

```python
# Persist in memory for repeated access
scene_persisted = scene.persist()

# Now operations are faster
result1 = scene_persisted.mean()
result2 = scene_persisted.std()
```

## Exercise

The `items` variable contains a list of STAC Items returned by the query. Extract the Sentinel-2 Product ID stored in `s2:product_uri` property and print a list of all image IDs.

```python
for item in items:
    print(item.properties)
```

**Solution:**

```python
product_ids = []
for item in items:
    product_id = item.properties.get('s2:product_uri', 'N/A')
    product_ids.append(product_id)
    print(product_id)

print(f"\nTotal images found: {len(product_ids)}")
```

## Key Takeaways

!!! success "What You Learned"
    - STAC provides a standardized way to discover geospatial data
    - Use `pystac-client` to search STAC catalogs
    - Filter by metadata (cloud cover, date, location)
    - Sort results to find optimal scenes
    - Dask enables parallel data loading and processing
    - Monitor Dask dashboard for performance insights
    - Load STAC items directly into XArray with `odc-stac`
    - Apply scale/offset to convert to physical values

## Next Steps

→ Continue to [Working with Zarr](zarr.md)

## Additional Resources

- [STAC Specification](https://stacspec.org/)
- [STAC Index](https://stacindex.org/)
- [pystac-client Documentation](https://pystac-client.readthedocs.io/)
- [odc-stac Documentation](https://odc-stac.readthedocs.io/)
- [Dask Documentation](https://docs.dask.org/)
- [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)
