# Datasets

## Overview

This page provides information about publicly available datasets for cloud-native remote sensing analysis.

## Satellite Imagery

### Sentinel-2

**Provider**: European Space Agency (ESA)  
**Resolution**: 10m, 20m, 60m  
**Revisit**: 5 days  
**Bands**: 13 spectral bands  
**Coverage**: Global  

**Access via STAC**:

```python
catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1')
search = catalog.search(
    collections=['sentinel-2-c1-l2a'],
    bbox=bbox,
    datetime='2023-01-01/2023-12-31'
)
```

**Direct Access**: [AWS Open Data](https://registry.opendata.aws/sentinel-2/)

### Landsat 8/9

**Provider**: USGS/NASA  
**Resolution**: 30m (15m panchromatic)  
**Revisit**: 16 days (8 days combined)  
**Bands**: 11 spectral bands  
**Coverage**: Global  

**Access via STAC**:

```python
search = catalog.search(
    collections=['landsat-c2-l2'],
    bbox=bbox,
    datetime='2023-01-01/2023-12-31'
)
```

**Direct Access**: [AWS Open Data](https://registry.opendata.aws/usgs-landsat/)

### MODIS

**Provider**: NASA  
**Resolution**: 250m, 500m, 1km  
**Revisit**: Daily  
**Products**: NDVI, LST, LAI, etc.  
**Coverage**: Global  

**Access via Earth Engine**:

```python
ds = xr.open_dataset(
    'ee://MODIS/006/MOD13A2',
    engine='ee',
    geometry=geometry,
    scale=1000
)
```

## Climate Data

### ERA5

**Provider**: ECMWF  
**Resolution**: ~25km  
**Temporal**: Hourly since 1940  
**Variables**: Temperature, precipitation, wind, etc.  

**Access**:

```python
ds = xr.open_dataset(
    'ee://ECMWF/ERA5/DAILY',
    engine='ee',
    geometry=geometry
)
```

### CHIRPS

**Provider**: UCSB/USGS  
**Resolution**: 5km  
**Temporal**: Daily since 1981  
**Variable**: Precipitation  

## Terrain Data

### SRTM

**Provider**: NASA/USGS  
**Resolution**: 30m, 90m  
**Coverage**: 60°N to 56°S  
**Product**: Digital Elevation Model  

**Access**:

```python
dem = xr.open_dataset(
    'ee://USGS/SRTMGL1_003',
    engine='ee',
    geometry=geometry
)
```

### ALOS World 3D

**Provider**: JAXA  
**Resolution**: 30m  
**Coverage**: Global  
**Product**: DSM, DTM  

## Land Cover

### ESA WorldCover

**Provider**: ESA  
**Resolution**: 10m  
**Year**: 2020, 2021  
**Classes**: 11 land cover types  

### Dynamic World

**Provider**: Google  
**Resolution**: 10m  
**Temporal**: 2015-present  
**Classes**: 9 land cover types  

## STAC Catalogs

### Earth Search (Element84)

- **URL**: <https://earth-search.aws.element84.com/v1>
- **Collections**: Sentinel-2, Landsat
- **Coverage**: Global
- **Free**: Yes

### Microsoft Planetary Computer

- **URL**: <https://planetarycomputer.microsoft.com/api/stac/v1>
- **Collections**: 50+ datasets
- **Coverage**: Global
- **Free**: Yes (registration required)

### Google Earth Engine

- **Collections**: 1000+ datasets
- **Access**: Via XEE
- **Free**: Yes (for research/education)

## Data Access Patterns

### Streaming (Recommended)

```python
# Don't download - stream from cloud
ds = xr.open_dataset('https://...', chunks='auto')
```

### Partial Downloads

```python
# Download only what you need
ds = xr.open_dataset('https://...').sel(
    time=slice('2023-01-01', '2023-01-31'),
    lat=slice(20, 30),
    lon=slice(70, 80)
)
```

### Zarr Archives

```python
# Efficient cloud access
ds = xr.open_zarr('s3://bucket/data.zarr')
```

## Storage Locations

### AWS S3

- Sentinel-2: `s3://sentinel-s2-l2a/`
- Landsat: `s3://usgs-landsat/`

### Google Cloud Storage

- Sentinel-2: `gs://gcp-public-data-sentinel-2/`
- Landsat: `gs://gcp-public-data-landsat/`

### Azure Blob Storage

- Sentinel-2: Available via Planetary Computer

## Best Practices

1. **Use STAC** for data discovery
2. **Stream data** instead of downloading
3. **Apply filters** server-side when possible
4. **Use appropriate resolution** for your analysis
5. **Leverage cloud-optimized formats** (COG, Zarr)

## Additional Resources

- [STAC Index](https://stacindex.org/)
- [AWS Open Data Registry](https://registry.opendata.aws/)
- [Google Earth Engine Catalog](https://developers.google.com/earth-engine/datasets)
- [Planetary Computer Catalog](https://planetarycomputer.microsoft.com/catalog)
