# Tools and Libraries

Essential tools and libraries for cloud-native remote sensing with Python.

## Core Libraries

### XArray

- **Purpose**: Multi-dimensional labeled arrays
- **Website**: [xarray.dev](https://xarray.dev)
- **Installation**: `pip install xarray`

### Dask

- **Purpose**: Parallel computing
- **Website**: [dask.org](https://dask.org)
- **Installation**: `pip install dask[complete]`

### Rioxarray

- **Purpose**: Geospatial extensions for XArray
- **Website**: [corteva.github.io/rioxarray](https://corteva.github.io/rioxarray)
- **Installation**: `pip install rioxarray`

## Data Access

### pystac-client

- **Purpose**: STAC catalog searching
- **Website**: [pystac-client.readthedocs.io](https://pystac-client.readthedocs.io)
- **Installation**: `pip install pystac-client`

### odc-stac

- **Purpose**: Load STAC items into XArray
- **Website**: [odc-stac.readthedocs.io](https://odc-stac.readthedocs.io)
- **Installation**: `pip install odc-stac`

### XEE

- **Purpose**: Earth Engine integration
- **Website**: [github.com/google/Xee](https://github.com/google/Xee)
- **Installation**: `pip install xee`

## Visualization

### Matplotlib

- **Purpose**: Static plotting
- **Installation**: `pip install matplotlib`

### Hvplot

- **Purpose**: Interactive visualizations
- **Installation**: `pip install hvplot`

### Geemap

- **Purpose**: Interactive Earth Engine maps
- **Installation**: `pip install geemap`

## Storage Formats

### Zarr

- **Purpose**: Cloud-optimized arrays
- **Installation**: `pip install zarr`

### NetCDF4

- **Purpose**: Self-describing data format
- **Installation**: `pip install netcdf4`

## Cloud Storage

### s3fs

- **Purpose**: AWS S3 filesystem
- **Installation**: `pip install s3fs`

### gcsfs

- **Purpose**: Google Cloud Storage
- **Installation**: `pip install gcsfs`

## Complete Installation

```bash
pip install xarray dask[complete] rioxarray pystac-client odc-stac xee \
    matplotlib hvplot geemap zarr netcdf4 s3fs gcsfs earthengine-api
```
