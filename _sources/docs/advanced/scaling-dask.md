# Scaling with Dask

When working with large-scale remote sensing data, a single machine's memory and computation power may not be enough. Dask allows you to scale your analysis from a single core on your laptop to a cluster of thousands of machines in the cloud.

## Why Dask for Remote Sensing?

- **Parallelism**: Process multiple image tiles or time steps simultaneously.
- **Out-of-core computing**: Work with datasets larger than your RAM by processing data in chunks.
- **XArray Integration**: XArray uses Dask as its primary engine for parallelized and lazy computation.

## Setting Up Dask

### Local Cluster

On a single machine, you can use a `LocalCluster` to utilize all available CPU cores.

```python
from dask.distributed import Client, LocalCluster

# Initialize a local cluster
cluster = LocalCluster()
client = Client(cluster)

# View the Dask dashboard link
print(f"Dask Dashboard: {client.dashboard_link}")
```

### Cloud Clusters

For massive processing, you can scale to cloud-based clusters like:

- **Dask Gateway**: Managed clusters in Kubernetes.
- **Coiled**: Managed Dask as a service.
- **Saturn Cloud**: Data science platform with built-in Dask support.

## Working with Chunks

The key to Dask performance is choosing the right chunk size. In remote sensing, this usually means chunking along the `time` dimension or `spatial` dimensions (X, Y).

```python
import xarray as xr

# Load data with automatic chunking
ds = xr.open_zarr('data.zarr', chunks='auto')

# Or specify manual chunks
ds = xr.open_zarr('data.zarr', chunks={'time': 10, 'x': 512, 'y': 512})
```

!!! tip "Chunk Size Rules"
    - **Too small**: High overhead, slow computation.
    - **Too large**: Memory errors (Dask needs to fit a few chunks in RAM per worker).
    - **Ideal**: Roughly 100MB - 200MB per chunk.

## Lazy Computation

Dask operations are "lazy" by default. This means they build a task graph but don't execute it until you explicitly request the result.

```python
# This builds the graph (no computation yet)
result = ds.NDVI.mean(dim='time')

# This triggers the parallel computation
final_mean = result.compute()
```

## Performance Monitoring

Dask provides a powerful dashboard that allows you to see:

- Worker memory usage
- Task progress
- CPU utilization per core
- Graph structure and potential bottlenecks

Using the dashboard is essential for optimizing your cloud-native remote sensing workflows.
