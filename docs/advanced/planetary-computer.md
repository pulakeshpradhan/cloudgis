# Microsoft Planetary Computer

The [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) is a flagship platform for cloud-native remote sensing. It combines a massive STAC-based data catalog with optimized computing resources.

## Core Components

### 1. Data Catalog

The Planetary Computer hosts petabytes of open geospatial data, including:

- **Sentinel-2 L2A** (harmonized)
- **Landsat 8 & 9** (Collection 2)
- **MODIS**
- **ERA5** Climate data
- **HLS** (Harmonized Landsat Sentinel)

All data is indexed via STAC, making it searchable using `pystac-client`.

### 2. Hub

A managed JupyterLab environment pre-configured with the geospatial stack (XArray, Dask, Stackstac, Riomente sensing tools).

### 3. API

A RESTful STAC API that allows you to search the catalog from any environment (Colab, local, or the Hub).

## Basic Usage

To access data from the Planetary Computer, you usually use `pystac_client` and `planetary_computer` for signing URLs.

```python
import pystac_client
import planetary_computer

# Connect to the API
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Sentinel-2 data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[12.4, 41.8, 12.5, 41.9],  # Rome coordinates
    datetime="2023-01-01/2023-12-31",
)

items = search.item_collection()
print(f"Found {len(items)} items")
```

## Why use Planetary Computer?

- **STAC First**: Everything is built around the STAC standard.
- **Signed URLs**: High-security access to data with "signing" that handles authentication automatically for you.
- **Pangeo Stack**: Built by and for the Pangeo community, emphasizing XArray and Dask.
- **Free for Research**: Most features are free for academic and research use.

## Comparison with Earth Engine

| Feature | Microsoft Planetary Computer | Google Earth Engine |
| --- | --- | --- |
| **Data Engine** | XArray / Dask (Python-native) | Earth Engine Object-based (JavaScript/Python API) |
| **API Style** | STAC / REST | Proprietary GEE API |
| **Philosophy** | Open-source ecosystem | Integrated SaaS platform |
| **Compute** | Managed Hub / Dask Gateway | Earth Engine EECU |

Both are excellent, but Planetary Computer is often preferred by those who want to use standard Python tools like XArray and Dask directly on the raw data.
