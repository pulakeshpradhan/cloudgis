# Zarr Format Reference

## Overview

Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. It's designed for use in parallel computing and cloud storage environments.

## Key Features

- **Chunked Storage**: Data divided into regular chunks
- **Compression**: Multiple compression algorithms supported
- **Cloud-Optimized**: Efficient for object storage (S3, GCS, Azure)
- **Parallel I/O**: Concurrent reads and writes
- **Language-Agnostic**: Implementations in Python, Julia, C++, Java
- **Metadata**: Flexible JSON metadata storage

## Storage Structure

### Directory Layout

```
data.zarr/
├── .zarray              # Array metadata
├── .zattrs              # User attributes
├── .zgroup              # Group metadata (if applicable)
└── 0.0.0                # Chunk files
    ├── 0.0.1
    ├── 0.1.0
    ├── 0.1.1
    └── ...
```

### Metadata Files

#### .zarray

```json
{
    "chunks": [10, 512, 512],
    "compressor": {
        "id": "blosc",
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 1
    },
    "dtype": "<f4",
    "fill_value": "NaN",
    "filters": null,
    "order": "C",
    "shape": [365, 5000, 5000],
    "zarr_format": 2
}
```

#### .zattrs

```json
{
    "title": "NDVI Time Series",
    "source": "Sentinel-2",
    "units": "dimensionless",
    "valid_range": [-1.0, 1.0]
}
```

## Chunking

### Chunk Size Selection

**Formula**: Aim for 10-100 MB per chunk

```python
import numpy as np

def optimal_chunks(shape, dtype, target_mb=10):
    """Calculate optimal chunk dimensions."""
    itemsize = np.dtype(dtype).itemsize
    target_items = (target_mb * 1024 * 1024) / itemsize
    
    # Distribute evenly across dimensions
    chunk_dim = int(target_items ** (1/len(shape)))
    chunks = tuple(min(chunk_dim, s) for s in shape)
    
    return chunks

# Example
shape = (365, 5000, 5000)
chunks = optimal_chunks(shape, 'float32', target_mb=10)
print(f"Recommended chunks: {chunks}")
```

### Access Pattern Optimization

#### Time Series Access

```python
# Optimize for temporal access
chunks = (1, 1000, 1000)  # Small time chunks, large spatial
```

#### Spatial Access

```python
# Optimize for spatial access
chunks = (365, 100, 100)  # Large time chunks, small spatial
```

#### Balanced Access

```python
# Balanced for mixed access
chunks = (10, 512, 512)
```

## Compression

### Blosc (Recommended)

Fast compression with multiple algorithms.

```python
from numcodecs import Blosc

# Fast compression (LZ4)
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)

# Balanced (Zstandard)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# High compression (Zstandard)
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)

# Use with Zarr
import zarr
z = zarr.open('data.zarr', mode='w', shape=(1000, 1000), 
              chunks=(100, 100), compressor=compressor)
```

### Other Compressors

```python
from numcodecs import Zlib, GZip, BZ2, LZMA

# Zlib (good compression, moderate speed)
compressor = Zlib(level=5)

# GZip (widely compatible)
compressor = GZip(level=6)

# BZ2 (high compression, slow)
compressor = BZ2(level=9)

# LZMA (highest compression, slowest)
compressor = LZMA(preset=6)
```

### Compression Comparison

| Compressor | Speed | Ratio | Use Case |
|------------|-------|-------|----------|
| LZ4 | Very Fast | Low | Real-time processing |
| Zstandard | Fast | Good | General purpose |
| Zlib | Medium | Good | Compatibility |
| BZ2 | Slow | High | Archival |
| LZMA | Very Slow | Highest | Long-term storage |

## XArray Integration

### Writing to Zarr

```python
import xarray as xr
from numcodecs import Blosc

# Create dataset
ds = xr.Dataset({
    'temperature': (['time', 'y', 'x'], data)
})

# Configure encoding
encoding = {
    'temperature': {
        'compressor': Blosc(cname='zstd', clevel=3),
        'chunks': (10, 512, 512)
    }
}

# Write to Zarr
ds.to_zarr('data.zarr', encoding=encoding, consolidated=True)
```

### Reading from Zarr

```python
# Open Zarr store
ds = xr.open_zarr('data.zarr', consolidated=True)

# With custom chunks
ds = xr.open_zarr('data.zarr', chunks={'time': 5})
```

### Appending Data

```python
# Initial write
ds1.to_zarr('timeseries.zarr', mode='w')

# Append along time dimension
ds2.to_zarr('timeseries.zarr', append_dim='time')
```

## Cloud Storage

### AWS S3

```python
import s3fs
import xarray as xr

# Create S3 filesystem
s3 = s3fs.S3FileSystem(anon=False)

# Write to S3
store = s3fs.S3Map(root='s3://my-bucket/data.zarr', s3=s3)
ds.to_zarr(store, mode='w', consolidated=True)

# Read from S3
ds = xr.open_zarr(store, consolidated=True)
```

### Google Cloud Storage

```python
import gcsfs

# Create GCS filesystem
gcs = gcsfs.GCSFileSystem(token='anon')

# Write to GCS
store = gcsfs.GCSMap('gs://my-bucket/data.zarr', gcs=gcs)
ds.to_zarr(store, mode='w', consolidated=True)

# Read from GCS
ds = xr.open_zarr(store, consolidated=True)
```

### Azure Blob Storage

```python
import adlfs

# Create Azure filesystem
fs = adlfs.AzureBlobFileSystem(account_name='myaccount')

# Write to Azure
store = fs.get_mapper('container/data.zarr')
ds.to_zarr(store, mode='w', consolidated=True)

# Read from Azure
ds = xr.open_zarr(store, consolidated=True)
```

## Consolidated Metadata

Improves performance by reducing number of reads.

### Create Consolidated Metadata

```python
# During write
ds.to_zarr('data.zarr', consolidated=True)

# After write
from zarr.convenience import consolidate_metadata
consolidate_metadata('data.zarr')
```

### Use Consolidated Metadata

```python
# Read with consolidated metadata (faster)
ds = xr.open_zarr('data.zarr', consolidated=True)

# Without consolidated metadata (slower)
ds = xr.open_zarr('data.zarr', consolidated=False)
```

## Parallel I/O

### Parallel Writing

```python
from dask.distributed import Client
import dask.array as da

client = Client()

# Create Dask array
data = da.random.random((1000, 5000, 5000), chunks=(10, 500, 500))

# Create dataset
ds = xr.Dataset({'data': (['time', 'y', 'x'], data)})

# Parallel write
ds.to_zarr('large_data.zarr', compute=True, consolidated=True)
```

### Parallel Reading

```python
# Open with Dask chunks
ds = xr.open_zarr('large_data.zarr', chunks={'time': 10})

# Operations are parallelized
result = ds.mean(dim='time').compute()
```

## Filters

Apply transformations before compression.

```python
from numcodecs import Delta, FixedScaleOffset

# Delta encoding (for time series)
filters = [Delta(dtype='i4')]

# Scale and offset
filters = [FixedScaleOffset(offset=0, scale=0.01, dtype='f4')]

# Use with Zarr
z = zarr.open('data.zarr', mode='w', shape=(1000,), 
              filters=filters, compressor=compressor)
```

## Groups and Hierarchies

### Creating Groups

```python
import zarr

# Create root group
root = zarr.open_group('data.zarr', mode='w')

# Create subgroups
temp_group = root.create_group('temperature')
precip_group = root.create_group('precipitation')

# Create arrays in groups
temp_group.create_dataset('daily', shape=(365, 100, 100), chunks=(10, 50, 50))
precip_group.create_dataset('daily', shape=(365, 100, 100), chunks=(10, 50, 50))
```

### Reading Groups

```python
# Open group
root = zarr.open_group('data.zarr', mode='r')

# Access subgroups
temp = root['temperature']
precip = root['precipitation']

# Access arrays
daily_temp = temp['daily']
```

## Best Practices

### 1. Use Consolidated Metadata

```python
# Always use for cloud storage
ds.to_zarr('s3://bucket/data.zarr', consolidated=True)
```

### 2. Choose Appropriate Chunks

```python
# Match access patterns
# Time series: large time chunks
chunks = (100, 512, 512)

# Spatial: large spatial chunks
chunks = (10, 1024, 1024)
```

### 3. Use Compression

```python
# Always compress for cloud storage
compressor = Blosc(cname='zstd', clevel=3)
```

### 4. Avoid Small Chunks

```python
# Good: ~10-100 MB per chunk
chunks = (10, 512, 512)  # ~10 MB for float32

# Bad: Too small
chunks = (1, 10, 10)  # ~400 bytes
```

### 5. Use Appropriate Data Types

```python
# Good: Use smallest appropriate dtype
ds = ds.astype('float32')

# Bad: Unnecessary precision
ds = ds.astype('float64')
```

## Performance Optimization

### Chunk Size Impact

```python
import time

# Test different chunk sizes
chunk_sizes = [(5, 256, 256), (10, 512, 512), (20, 1024, 1024)]

for chunks in chunk_sizes:
    start = time.time()
    ds.chunk(chunks).to_zarr(f'test_{chunks[0]}.zarr', mode='w')
    write_time = time.time() - start
    
    start = time.time()
    loaded = xr.open_zarr(f'test_{chunks[0]}.zarr').compute()
    read_time = time.time() - start
    
    print(f"Chunks {chunks}: Write={write_time:.2f}s, Read={read_time:.2f}s")
```

### Compression Impact

```python
# Test compression algorithms
compressors = {
    'none': None,
    'lz4': Blosc(cname='lz4', clevel=5),
    'zstd': Blosc(cname='zstd', clevel=3),
    'zlib': Zlib(level=5)
}

for name, comp in compressors.items():
    encoding = {'data': {'compressor': comp}}
    ds.to_zarr(f'test_{name}.zarr', encoding=encoding, mode='w')
    
    # Check size
    size = sum(f.stat().st_size for f in Path(f'test_{name}.zarr').rglob('*'))
    print(f"{name}: {size/1e6:.2f} MB")
```

## Troubleshooting

### Issue: Slow Reads

```python
# Solution 1: Use consolidated metadata
ds = xr.open_zarr('data.zarr', consolidated=True)

# Solution 2: Check chunk size
print(ds.chunks)

# Solution 3: Use appropriate chunks for access pattern
ds = xr.open_zarr('data.zarr', chunks={'time': 10})
```

### Issue: Large File Size

```python
# Solution: Use compression
encoding = {
    'var': {'compressor': Blosc(cname='zstd', clevel=5)}
}
ds.to_zarr('data.zarr', encoding=encoding)
```

### Issue: Memory Errors

```python
# Solution: Use smaller chunks
ds = xr.open_zarr('data.zarr', chunks={'time': 1})
```

## Additional Resources

- [Zarr Documentation](https://zarr.readthedocs.io/)
- [Zarr Tutorial](https://zarr.readthedocs.io/en/stable/tutorial.html)
- [Zarr Specification](https://zarr-specs.readthedocs.io/)
- [XArray Zarr Guide](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr)
- [Pangeo Zarr Examples](https://pangeo.io/data.html#zarr)

## Quick Reference

### Common Operations

```python
import zarr
import xarray as xr

# Create Zarr array
z = zarr.open('data.zarr', mode='w', shape=(1000, 1000), 
              chunks=(100, 100), dtype='f4')

# Write XArray to Zarr
ds.to_zarr('data.zarr', mode='w', consolidated=True)

# Read Zarr with XArray
ds = xr.open_zarr('data.zarr', consolidated=True)

# Append data
ds.to_zarr('data.zarr', append_dim='time')

# Consolidate metadata
from zarr.convenience import consolidate_metadata
consolidate_metadata('data.zarr')
```

### Recommended Settings

```python
# General purpose
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
chunks = (10, 512, 512)  # For (time, y, x)

# High compression (archival)
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)

# Fast I/O (real-time)
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
```
