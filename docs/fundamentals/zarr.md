# Working with Zarr

## Overview

[Zarr](https://zarr.readthedocs.io/) is a cloud-optimized format for storing chunked, compressed N-dimensional arrays. It's designed for efficient storage and access of large scientific datasets, making it ideal for remote sensing applications.

**Key Features:**

- Cloud-native storage format
- Chunked and compressed arrays
- Parallel read/write operations
- Works seamlessly with Dask and XArray
- Supports multiple storage backends (local, S3, GCS, Azure)

## Why Zarr?

### Traditional Formats (NetCDF, HDF5)

- Designed for local filesystems
- Poor performance over HTTP
- Sequential access patterns
- Difficult to parallelize

### Zarr Advantages

- ✅ Optimized for cloud storage
- ✅ Parallel read/write
- ✅ Efficient partial reads
- ✅ Multiple compression algorithms
- ✅ Language-agnostic specification

## Basic Zarr Operations

### Creating a Zarr Array

```python
import zarr
import numpy as np

# Create a Zarr array
z = zarr.open('data.zarr', mode='w', shape=(10000, 10000), 
              chunks=(1000, 1000), dtype='f4', 
              compressor=zarr.Blosc(cname='zstd', clevel=3))

# Write data
data = np.random.random((10000, 10000))
z[:] = data
```

### Reading Zarr Data

```python
# Open existing Zarr array
z = zarr.open('data.zarr', mode='r')

# Read subset
subset = z[1000:2000, 1000:2000]

# Read entire array
all_data = z[:]
```

### Zarr with XArray

```python
import xarray as xr

# Create XArray dataset
ds = xr.Dataset({
    'temperature': (['time', 'y', 'x'], np.random.random((365, 1000, 1000))),
    'precipitation': (['time', 'y', 'x'], np.random.random((365, 1000, 1000)))
})

# Save to Zarr
ds.to_zarr('climate_data.zarr', mode='w')

# Load from Zarr
ds_loaded = xr.open_zarr('climate_data.zarr')
```

## Chunking Strategies

Chunking is critical for performance. Choose chunk sizes based on your access patterns.

### Time-Series Access

If you frequently access time slices:

```python
# Optimize for time-series access
ds.to_zarr('timeseries.zarr', 
           encoding={
               'temperature': {'chunks': (1, 1000, 1000)}
           })
```

### Spatial Access

If you frequently access spatial subsets:

```python
# Optimize for spatial access
ds.to_zarr('spatial.zarr',
           encoding={
               'temperature': {'chunks': (365, 100, 100)}
           })
```

### Balanced Chunking

For mixed access patterns:

```python
# Balanced chunks
ds.to_zarr('balanced.zarr',
           encoding={
               'temperature': {'chunks': (10, 512, 512)}
           })
```

### Chunk Size Guidelines

!!! tip "Optimal Chunk Sizes"
    - **Minimum**: 1 MB per chunk
    - **Maximum**: 100 MB per chunk
    - **Optimal**: 10-50 MB per chunk
    - **Rule of thumb**: Aim for ~10,000 chunks total

```python
# Calculate chunk size
import numpy as np

def calculate_chunk_size(shape, dtype, target_mb=10):
    """Calculate optimal chunk dimensions."""
    itemsize = np.dtype(dtype).itemsize
    target_bytes = target_mb * 1024 * 1024
    total_items = target_bytes / itemsize
    
    # Distribute across dimensions
    chunk_dim = int(total_items ** (1/len(shape)))
    chunks = tuple(min(chunk_dim, s) for s in shape)
    
    return chunks

# Example
shape = (365, 5000, 5000)
chunks = calculate_chunk_size(shape, 'float32', target_mb=10)
print(f"Recommended chunks: {chunks}")
```

## Compression

Zarr supports multiple compression algorithms:

### Blosc (Recommended)

```python
from zarr import Blosc

# Fast compression
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)

# Balanced
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# High compression
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)

# Use with XArray
ds.to_zarr('compressed.zarr',
           encoding={
               'temperature': {'compressor': compressor}
           })
```

### Other Compressors

```python
from numcodecs import Zlib, GZip, BZ2, LZMA

# Zlib (good compression)
compressor = Zlib(level=5)

# GZip (compatible)
compressor = GZip(level=6)

# LZMA (high compression, slow)
compressor = LZMA(preset=6)
```

### Compression Comparison

```python
import time

compressors = {
    'none': None,
    'lz4': Blosc(cname='lz4', clevel=5),
    'zstd': Blosc(cname='zstd', clevel=3),
    'zlib': Zlib(level=5)
}

data = np.random.random((1000, 1000, 100))

for name, comp in compressors.items():
    start = time.time()
    z = zarr.open(f'test_{name}.zarr', mode='w',
                  shape=data.shape, chunks=(100, 100, 10),
                  compressor=comp)
    z[:] = data
    write_time = time.time() - start
    
    size = sum(f.stat().st_size for f in Path(f'test_{name}.zarr').rglob('*') if f.is_file())
    
    print(f"{name:10s} - Size: {size/1e6:6.2f} MB, Time: {write_time:5.2f}s")
```

## Cloud Storage

### AWS S3

```python
import s3fs

# Create S3 filesystem
s3 = s3fs.S3FileSystem(anon=False)

# Write to S3
store = s3fs.S3Map(root='s3://my-bucket/data.zarr', s3=s3)
ds.to_zarr(store, mode='w')

# Read from S3
ds_s3 = xr.open_zarr(store)
```

### Google Cloud Storage

```python
import gcsfs

# Create GCS filesystem
gcs = gcsfs.GCSFileSystem(token='anon')

# Write to GCS
store = gcsfs.GCSMap('gs://my-bucket/data.zarr', gcs=gcs)
ds.to_zarr(store, mode='w')

# Read from GCS
ds_gcs = xr.open_zarr(store)
```

### Azure Blob Storage

```python
import adlfs

# Create Azure filesystem
fs = adlfs.AzureBlobFileSystem(account_name='myaccount')

# Write to Azure
store = fs.get_mapper('container/data.zarr')
ds.to_zarr(store, mode='w')

# Read from Azure
ds_azure = xr.open_zarr(store)
```

## Appending Data

Zarr supports appending along dimensions:

```python
# Initial dataset
ds1 = xr.Dataset({
    'temperature': (['time', 'y', 'x'], np.random.random((10, 100, 100)))
})
ds1.to_zarr('timeseries.zarr', mode='w')

# Append new time steps
ds2 = xr.Dataset({
    'temperature': (['time', 'y', 'x'], np.random.random((5, 100, 100)))
})
ds2.to_zarr('timeseries.zarr', append_dim='time')

# Load combined dataset
ds_combined = xr.open_zarr('timeseries.zarr')
print(ds_combined.dims)  # time: 15
```

## Parallel Writing

Use Dask for parallel writes:

```python
from dask.distributed import Client

client = Client()

# Create large dataset with Dask
ds_large = xr.Dataset({
    'data': (['time', 'y', 'x'], 
             da.random.random((1000, 5000, 5000), chunks=(10, 500, 500)))
})

# Parallel write to Zarr
ds_large.to_zarr('large_data.zarr', 
                 compute=True,
                 consolidated=True)
```

## Metadata and Attributes

### Store Metadata

```python
# Add attributes
ds.attrs['title'] = 'Climate Data'
ds.attrs['source'] = 'Satellite Observations'
ds.attrs['processing_date'] = '2024-01-01'

# Variable attributes
ds['temperature'].attrs['units'] = 'Kelvin'
ds['temperature'].attrs['long_name'] = 'Air Temperature'

# Save with metadata
ds.to_zarr('data_with_metadata.zarr')
```

### Consolidated Metadata

Improve performance with consolidated metadata:

```python
# Write with consolidated metadata
ds.to_zarr('data.zarr', consolidated=True)

# Or consolidate existing Zarr
from zarr.convenience import consolidate_metadata
consolidate_metadata('data.zarr')

# Read with consolidated metadata (faster)
ds = xr.open_zarr('data.zarr', consolidated=True)
```

## Real-World Example: Sentinel-2 Time Series

```python
import pystac_client
from odc.stac import load as stac_load

# Search for Sentinel-2 data
catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1')

search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=[lon_min, lat_min, lon_max, lat_max],
    datetime='2023-01-01/2023-12-31',
    query={'eo:cloud_cover': {'lt': 30}}
)
items = search.item_collection()

# Load as XArray with Dask
ds = stac_load(
    items,
    bands=['red', 'green', 'blue', 'nir'],
    resolution=10,
    chunks={'time': 10, 'x': 512, 'y': 512}
)

# Calculate NDVI
ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)

# Save to Zarr with compression
encoding = {
    'ndvi': {
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'chunks': (10, 512, 512)
    }
}

ndvi.to_dataset(name='ndvi').to_zarr(
    'sentinel2_ndvi_2023.zarr',
    encoding=encoding,
    consolidated=True
)

# Load and analyze
ndvi_loaded = xr.open_zarr('sentinel2_ndvi_2023.zarr')
monthly_mean = ndvi_loaded.resample(time='1M').mean()
```

## Performance Tips

### 1. Use Consolidated Metadata

```python
# Always use consolidated metadata for cloud storage
ds.to_zarr('data.zarr', consolidated=True)
```

### 2. Choose Appropriate Chunks

```python
# Match chunks to access patterns
# Time-series: large time chunks
# Spatial: large spatial chunks
```

### 3. Use Compression

```python
# Blosc with zstd is usually best
compressor = zarr.Blosc(cname='zstd', clevel=3)
```

### 4. Parallel I/O

```python
# Use Dask for parallel operations
ds.to_zarr('data.zarr', compute=True)
```

### 5. Avoid Small Chunks

```python
# Bad: too many small chunks
chunks = (1, 10, 10)  # Only 100 items per chunk

# Good: reasonable chunk size
chunks = (10, 512, 512)  # ~2.6M items per chunk
```

## Troubleshooting

### Issue: Slow Reads

```python
# Solution: Check chunk size and use consolidated metadata
ds = xr.open_zarr('data.zarr', consolidated=True)
print(ds.chunks)
```

### Issue: Large File Size

```python
# Solution: Use compression
ds.to_zarr('data.zarr', 
           encoding={'var': {'compressor': zarr.Blosc(cname='zstd', clevel=5)}})
```

### Issue: Memory Errors

```python
# Solution: Use smaller chunks
ds = xr.open_zarr('data.zarr', chunks={'time': 1})
```

## Key Takeaways

!!! success "What You Learned"
    - Zarr is optimized for cloud storage and parallel access
    - Chunking strategy depends on access patterns
    - Compression reduces storage costs
    - Consolidated metadata improves performance
    - Zarr works seamlessly with XArray and Dask
    - Supports appending and parallel writes
    - Multiple cloud storage backends supported

## Next Steps

→ Continue to [XEE for Earth Engine](xee.md)

## Additional Resources

- [Zarr Documentation](https://zarr.readthedocs.io/)
- [Zarr Tutorial](https://zarr.readthedocs.io/en/stable/tutorial.html)
- [XArray Zarr Guide](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr)
- [Pangeo Zarr Guide](https://pangeo.io/data.html#zarr)
- [Cloud-Optimized Formats](https://guide.cloudnativegeo.org/)
