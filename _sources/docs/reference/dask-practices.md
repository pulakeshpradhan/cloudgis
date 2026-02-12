# Dask Best Practices

## Overview

Dask is a flexible parallel computing library for Python that scales from laptops to clusters. It's particularly useful for processing large remote sensing datasets that don't fit in memory.

## Core Concepts

### Lazy Evaluation

Dask operations are lazy - they build a task graph without executing until you call `.compute()`.

```python
import dask.array as da
import xarray as xr

# This doesn't load data
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# This builds a task graph (still lazy)
result = ds.mean(dim='time')

# This triggers computation
result_computed = result.compute()
```

### Task Graph

Dask builds a graph of operations to optimize execution.

```python
# View task graph
result.data.visualize(filename='task_graph.png')
```

## Chunking Strategies

### Chunk Size Guidelines

**Optimal chunk size**: 10-100 MB per chunk

```python
# Calculate chunk size
import numpy as np

def calculate_chunk_size_mb(shape, dtype, chunks):
    """Calculate chunk size in MB."""
    itemsize = np.dtype(dtype).itemsize
    chunk_items = np.prod([chunks[i] if i < len(chunks) else shape[i] 
                           for i in range(len(shape))])
    return (chunk_items * itemsize) / 1e6

# Example
shape = (365, 5000, 5000)
chunks = (10, 500, 500)
size_mb = calculate_chunk_size_mb(shape, 'float32', chunks)
print(f"Chunk size: {size_mb:.2f} MB")
```

### Time Series Data

```python
# Good: Large time chunks for temporal operations
ds = xr.open_dataset('timeseries.nc', chunks={'time': 100, 'x': 512, 'y': 512})

# Bad: Small time chunks
ds = xr.open_dataset('timeseries.nc', chunks={'time': 1, 'x': 512, 'y': 512})
```

### Spatial Data

```python
# Good: Balanced spatial chunks
ds = xr.open_dataset('spatial.nc', chunks={'time': 10, 'x': 512, 'y': 512})

# Bad: Unbalanced chunks
ds = xr.open_dataset('spatial.nc', chunks={'time': 10, 'x': 5000, 'y': 10})
```

### Auto Chunking

```python
# Let Dask decide
ds = xr.open_dataset('file.nc', chunks='auto')

# With target chunk size
ds = xr.open_zarr('data.zarr', chunks={'time': 'auto'})
```

## Dask Schedulers

### Single Machine (Threaded)

Default scheduler, good for I/O-bound tasks.

```python
# Automatic (default)
result = ds.mean().compute()

# Explicit
result = ds.mean().compute(scheduler='threads')
```

### Single Machine (Processes)

Better for CPU-bound tasks, avoids GIL.

```python
result = ds.mean().compute(scheduler='processes')
```

### Distributed

Best for large computations, provides dashboard.

```python
from dask.distributed import Client, LocalCluster

# Start local cluster
cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit='4GB'
)
client = Client(cluster)

# Computations use distributed scheduler
result = ds.mean().compute()

# Close when done
client.close()
cluster.close()
```

## Memory Management

### Persist vs Compute

```python
# compute(): Load into memory as numpy/pandas
result = ds.mean().compute()

# persist(): Keep as Dask array in distributed memory
result = ds.mean().persist()

# Use persist() for intermediate results
intermediate = ds.resample(time='1M').mean().persist()
result1 = intermediate.max()
result2 = intermediate.min()
```

### Clear Memory

```python
# Delete large objects
del large_array

# Clear Dask cache
from dask import config
config.set(scheduler='synchronous')
```

## Optimization Techniques

### 1. Rechunking

```python
# Rechunk for different access patterns
ds_rechunked = ds.chunk({'time': 1, 'x': 1000, 'y': 1000})

# Optimize chunks
from dask.array import rechunk
optimized = rechunk(ds.data, chunks=(10, 512, 512))
```

### 2. Avoid Small Tasks

```python
# Good: Reasonable chunk size
ds = ds.chunk({'time': 10, 'x': 512, 'y': 512})

# Bad: Too many small tasks
ds = ds.chunk({'time': 1, 'x': 10, 'y': 10})
```

### 3. Use map_blocks

```python
def process_block(block):
    # Custom processing
    return block * 2 + 10

result = ds.map_blocks(process_block, dtype=float)
```

### 4. Avoid Repeated Computation

```python
# Good: Compute once
mean = ds.mean(dim='time').persist()
result1 = mean + 10
result2 = mean * 2

# Bad: Recompute each time
result1 = ds.mean(dim='time') + 10
result2 = ds.mean(dim='time') * 2
```

## Distributed Computing

### Local Cluster

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit='2GB',
    processes=True
)
client = Client(cluster)

# View dashboard
print(client.dashboard_link)
```

### Cluster Configuration

```python
# Custom worker configuration
cluster = LocalCluster(
    n_workers=8,
    threads_per_worker=1,
    memory_limit='4GB',
    processes=True,
    silence_logs=False
)
```

### Adaptive Scaling

```python
# Auto-scale workers
cluster.adapt(minimum=2, maximum=10)
```

## Monitoring

### Dashboard

```python
# Access dashboard
print(client.dashboard_link)
# Usually: http://localhost:8787/status
```

### Progress Bar

```python
from dask.diagnostics import ProgressBar

with ProgressBar():
    result = ds.mean().compute()
```

### Performance Report

```python
from dask.distributed import performance_report

with performance_report(filename='dask-report.html'):
    result = ds.mean().compute()
```

## Common Patterns

### Time Series Processing

```python
# Monthly aggregation
monthly = ds.resample(time='1M').mean()

# With Dask
monthly = ds.chunk({'time': 30}).resample(time='1M').mean().compute()
```

### Spatial Operations

```python
# Spatial mean
spatial_mean = ds.mean(dim=['x', 'y'])

# With Dask
spatial_mean = ds.chunk({'x': 512, 'y': 512}).mean(dim=['x', 'y']).compute()
```

### Rolling Windows

```python
# 7-day rolling mean
rolling = ds.rolling(time=7, center=True).mean()

# With Dask
rolling = ds.chunk({'time': 30}).rolling(time=7, center=True).mean().compute()
```

### GroupBy Operations

```python
# Monthly climatology
monthly_clim = ds.groupby('time.month').mean()

# With Dask
monthly_clim = ds.chunk({'time': 30}).groupby('time.month').mean().compute()
```

## Troubleshooting

### Memory Errors

```python
# Solution 1: Smaller chunks
ds = ds.chunk({'time': 5, 'x': 256, 'y': 256})

# Solution 2: More workers with less memory each
cluster = LocalCluster(n_workers=8, memory_limit='1GB')

# Solution 3: Process in batches
for year in range(2020, 2024):
    subset = ds.sel(time=str(year))
    result = subset.mean().compute()
```

### Slow Performance

```python
# Solution 1: Check chunk size
print(ds.chunks)

# Solution 2: Use distributed scheduler
client = Client()

# Solution 3: Persist intermediate results
intermediate = ds.mean(dim='time').persist()
```

### Too Many Tasks

```python
# Solution: Increase chunk size
ds = ds.chunk({'time': 50, 'x': 1024, 'y': 1024})
```

## Best Practices Summary

### ✅ Do

1. **Use appropriate chunk sizes** (10-100 MB)
2. **Persist intermediate results** used multiple times
3. **Use distributed scheduler** for large computations
4. **Monitor with dashboard**
5. **Close clients and clusters** when done
6. **Rechunk for access patterns**
7. **Use map_blocks for custom functions**

### ❌ Don't

1. **Create too many small chunks**
2. **Call compute() repeatedly** on same data
3. **Mix schedulers** in same script
4. **Forget to close** distributed clients
5. **Use tiny chunk sizes** (< 1 MB)
6. **Use huge chunk sizes** (> 1 GB)
7. **Ignore memory limits**

## Configuration

### Global Configuration

```python
import dask

# Set scheduler
dask.config.set(scheduler='threads')

# Set chunk size
dask.config.set({'array.chunk-size': '128 MiB'})

# Disable task fusion
dask.config.set({'optimization.fuse.active': False})
```

### Environment Variables

```bash
# Set number of threads
export OMP_NUM_THREADS=4

# Set Dask config directory
export DASK_CONFIG=/path/to/config
```

## Performance Tips

### 1. Optimize I/O

```python
# Good: Read with chunks
ds = xr.open_zarr('data.zarr', chunks='auto')

# Bad: Read without chunks
ds = xr.open_zarr('data.zarr')
```

### 2. Use Efficient Formats

```python
# Good: Zarr (cloud-optimized)
ds.to_zarr('output.zarr')

# OK: NetCDF with compression
ds.to_netcdf('output.nc', encoding={'var': {'zlib': True}})

# Bad: Uncompressed NetCDF
ds.to_netcdf('output.nc')
```

### 3. Minimize Data Transfer

```python
# Good: Reduce before computing
result = ds.mean(dim=['x', 'y']).compute()

# Bad: Compute then reduce
result = ds.compute().mean(dim=['x', 'y'])
```

### 4. Use Appropriate Data Types

```python
# Good: Use smaller dtypes when possible
ds = ds.astype('float32')

# Bad: Unnecessary precision
ds = ds.astype('float64')
```

## Advanced Topics

### Custom Schedulers

```python
from dask.threaded import get

result = ds.mean().compute(scheduler=get)
```

### Task Priorities

```python
# High priority tasks
result = ds.mean().compute(priority=10)
```

### Resource Management

```python
cluster = LocalCluster(
    n_workers=4,
    resources={'GPU': 1}
)

# Use resources
result = ds.map_blocks(gpu_function, resources={'GPU': 1})
```

## Additional Resources

- [Dask Documentation](https://docs.dask.org/)
- [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)
- [Dask Tutorial](https://tutorial.dask.org/)
- [Dask Examples](https://examples.dask.org/)
- [Pangeo Dask Guide](https://pangeo.io/packages.html#dask)

## Quick Reference

### Common Operations

```python
# Load with chunks
ds = xr.open_dataset('file.nc', chunks={'time': 10})

# Compute
result = ds.mean().compute()

# Persist
result = ds.mean().persist()

# Rechunk
ds = ds.chunk({'time': 20})

# Start client
from dask.distributed import Client
client = Client()

# Close client
client.close()
```

### Chunk Size Calculation

```python
# Target: 10-100 MB per chunk
# Formula: chunk_size = (chunk_items * itemsize) / 1e6

# Example for float32 (4 bytes)
# 10 MB: ~2.5M items
# 100 MB: ~25M items

# For (time, x, y) = (10, 500, 500)
# Items = 10 * 500 * 500 = 2.5M
# Size = 2.5M * 4 bytes = 10 MB ✓
```
