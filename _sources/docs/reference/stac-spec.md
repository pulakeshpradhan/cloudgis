# STAC Specification Reference

## Overview

The **SpatioTemporal Asset Catalog (STAC)** specification provides a common language to describe geospatial information, making it easier to index, discover, and access satellite imagery and other geospatial assets.

## Core Components

### 1. STAC Item

A STAC Item is a GeoJSON Feature with additional fields that describe a spatiotemporal asset.

**Structure:**

```json
{
  "type": "Feature",
  "stac_version": "1.0.0",
  "stac_extensions": [],
  "id": "S2A_MSIL2A_20230115T051131_N0509_R019_T44QMG_20230115T073857",
  "bbox": [82.0, 26.5, 83.0, 27.5],
  "geometry": {
    "type": "Polygon",
    "coordinates": [[...]]
  },
  "properties": {
    "datetime": "2023-01-15T05:11:31Z",
    "eo:cloud_cover": 15.5,
    "platform": "sentinel-2a",
    "instruments": ["msi"]
  },
  "assets": {
    "red": {
      "href": "s3://sentinel-s2-l2a/.../B04.jp2",
      "type": "image/jp2",
      "eo:bands": [{
        "name": "B04",
        "common_name": "red",
        "center_wavelength": 0.665
      }]
    },
    "nir": {
      "href": "s3://sentinel-s2-l2a/.../B08.jp2",
      "type": "image/jp2",
      "eo:bands": [{
        "name": "B08",
        "common_name": "nir",
        "center_wavelength": 0.842
      }]
    }
  },
  "links": [
    {
      "rel": "self",
      "href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/..."
    },
    {
      "rel": "parent",
      "href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
    }
  ]
}
```

**Key Fields:**

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique identifier for the item |
| `bbox` | array | Bounding box [west, south, east, north] |
| `geometry` | GeoJSON | Spatial extent as GeoJSON geometry |
| `properties` | object | Additional metadata (datetime, cloud cover, etc.) |
| `assets` | object | Links to actual data files |
| `links` | array | Relationships to other STAC objects |

### 2. STAC Collection

A STAC Collection provides metadata about a set of related Items.

**Structure:**

```json
{
  "type": "Collection",
  "stac_version": "1.0.0",
  "id": "sentinel-2-l2a",
  "title": "Sentinel-2 Level-2A",
  "description": "Global Sentinel-2 data from the Multispectral Instrument (MSI)",
  "license": "proprietary",
  "extent": {
    "spatial": {
      "bbox": [[-180, -90, 180, 90]]
    },
    "temporal": {
      "interval": [["2015-06-27T00:00:00Z", null]]
    }
  },
  "summaries": {
    "platform": ["sentinel-2a", "sentinel-2b"],
    "instruments": ["msi"],
    "eo:cloud_cover": {
      "minimum": 0,
      "maximum": 100
    }
  },
  "links": [...]
}
```

**Key Fields:**

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Collection identifier |
| `title` | string | Human-readable title |
| `description` | string | Detailed description |
| `extent` | object | Spatial and temporal extent |
| `summaries` | object | Summary statistics of properties |
| `license` | string | Data license |

### 3. STAC Catalog

A STAC Catalog is a simple, flexible JSON file of links to Items, Collections, and other Catalogs.

**Structure:**

```json
{
  "type": "Catalog",
  "stac_version": "1.0.0",
  "id": "earth-search",
  "title": "Earth Search",
  "description": "A STAC API of public datasets on AWS",
  "links": [
    {
      "rel": "self",
      "href": "https://earth-search.aws.element84.com/v1"
    },
    {
      "rel": "child",
      "href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
    },
    {
      "rel": "child",
      "href": "https://earth-search.aws.element84.com/v1/collections/landsat-c2-l2"
    }
  ]
}
```

## STAC API

The STAC API specification defines a RESTful service for searching STAC Items.

### Core Endpoints

#### 1. Landing Page

```http
GET /
```

Returns the catalog root with links to collections and search.

#### 2. Collections

```http
GET /collections
GET /collections/{collectionId}
```

List all collections or get a specific collection.

#### 3. Items

```http
GET /collections/{collectionId}/items
GET /collections/{collectionId}/items/{itemId}
```

List items in a collection or get a specific item.

#### 4. Search

```http
POST /search
GET /search
```

Search for items across collections.

### Search Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `bbox` | array | Bounding box [west, south, east, north] |
| `datetime` | string | Single datetime or range (RFC 3339) |
| `intersects` | GeoJSON | GeoJSON geometry to search |
| `collections` | array | Collection IDs to search |
| `ids` | array | Specific item IDs |
| `limit` | integer | Maximum number of results |
| `query` | object | Additional property filters |
| `sortby` | array | Sort results by fields |

### Search Examples

#### Basic Spatial Search

```python
import pystac_client

catalog = pystac_client.Client.open(
    'https://earth-search.aws.element84.com/v1'
)

search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[82.0, 26.5, 83.0, 27.5]
)

items = search.item_collection()
print(f"Found {len(items)} items")
```

#### Temporal Search

```python
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[82.0, 26.5, 83.0, 27.5],
    datetime='2023-01-01/2023-12-31'
)
```

#### Property Filtering

```python
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[82.0, 26.5, 83.0, 27.5],
    datetime='2023-01-01/2023-12-31',
    query={
        'eo:cloud_cover': {'lt': 20},
        's2:nodata_pixel_percentage': {'lt': 10}
    }
)
```

#### Geometry Search

```python
from shapely.geometry import Point

point = Point(82.5, 27.0)
buffer = point.buffer(0.1)

search = catalog.search(
    collections=['sentinel-2-l2a'],
    intersects=buffer.__geo_interface__,
    datetime='2023-01-01/2023-12-31'
)
```

#### Sorting Results

```python
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[82.0, 26.5, 83.0, 27.5],
    datetime='2023-01-01/2023-12-31',
    sortby=[
        {'field': 'properties.eo:cloud_cover', 'direction': 'asc'},
        {'field': 'properties.datetime', 'direction': 'desc'}
    ]
)
```

## STAC Extensions

STAC Extensions add additional fields and functionality.

### Common Extensions

#### 1. EO (Electro-Optical)

Adds fields for optical satellite data.

**Fields:**

- `eo:cloud_cover` - Cloud coverage percentage
- `eo:bands` - Band information (name, wavelength, etc.)

#### 2. SAR (Synthetic Aperture Radar)

For radar satellite data.

**Fields:**

- `sar:instrument_mode` - Instrument mode (IW, EW, etc.)
- `sar:frequency_band` - Frequency band (C, X, L, etc.)
- `sar:polarizations` - Polarization modes (VV, VH, HH, HV)

#### 3. Projection

Coordinate reference system information.

**Fields:**

- `proj:epsg` - EPSG code
- `proj:wkt2` - WKT2 projection string
- `proj:shape` - Raster dimensions
- `proj:transform` - Affine transformation

#### 4. Scientific

For scientific datasets.

**Fields:**

- `sci:doi` - Digital Object Identifier
- `sci:citation` - Citation string
- `sci:publications` - Related publications

## Best Practices

### 1. Use Appropriate Filters

```python
# Good: Filter server-side
search = catalog.search(
    collections=['sentinel-2-l2a'],
    query={'eo:cloud_cover': {'lt': 20}}
)

# Bad: Filter client-side
search = catalog.search(collections=['sentinel-2-l2a'])
items = [item for item in search.items() if item.properties['eo:cloud_cover'] < 20]
```

### 2. Limit Results

```python
# Always set a reasonable limit
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    limit=100
)
```

### 3. Use Pagination

```python
# Handle large result sets
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=large_bbox,
    max_items=1000  # pystac-client handles pagination
)

for item in search.items():
    process(item)
```

### 4. Check Item Properties

```python
# Always check if property exists
cloud_cover = item.properties.get('eo:cloud_cover', None)
if cloud_cover is not None and cloud_cover < 20:
    process(item)
```

## Public STAC Catalogs

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

- **Access**: Via XEE
- **Collections**: 1000+ datasets
- **Coverage**: Global
- **Free**: Yes (for research/education)

### Radiant Earth MLHub

- **URL**: <https://api.radiant.earth/mlhub/v1>
- **Collections**: Training datasets
- **Coverage**: Various
- **Free**: Yes

## Tools and Libraries

### Python

- **pystac** - Create and manipulate STAC objects
- **pystac-client** - Search STAC APIs
- **odc-stac** - Load STAC items into XArray
- **stackstac** - Load STAC items into Dask arrays

### JavaScript

- **stac-js** - STAC utilities
- **@radiantearth/stac-browser** - Browse STAC catalogs

### Command Line

- **stac-cli** - Command-line STAC tools

## Additional Resources

- [STAC Specification](https://stacspec.org/)
- [STAC Index](https://stacindex.org/)
- [STAC Extensions](https://stac-extensions.github.io/)
- [pystac Documentation](https://pystac.readthedocs.io/)
- [pystac-client Documentation](https://pystac-client.readthedocs.io/)
- [STAC Tutorials](https://stacspec.org/en/tutorials/)

## Quick Reference

### Common Query Operators

| Operator | Description | Example |
| --- | --- | --- |
| `eq` | Equal to | `{'platform': {'eq': 'sentinel-2a'}}` |
| `lt` | Less than | `{'eo:cloud_cover': {'lt': 20}}` |
| `lte` | Less than or equal | `{'eo:cloud_cover': {'lte': 20}}` |
| `gt` | Greater than | `{'eo:cloud_cover': {'gt': 50}}` |
| `gte` | Greater than or equal | `{'eo:cloud_cover': {'gte': 50}}` |
| `in` | In list | `{'platform': {'in': ['sentinel-2a', 'sentinel-2b']}}` |

### Datetime Formats

```python
# Single datetime
datetime='2023-01-15T10:30:00Z'

# Date range
datetime='2023-01-01/2023-12-31'

# Open-ended (from date)
datetime='2023-01-01/..'

# Open-ended (to date)
datetime='../2023-12-31'
```

### Link Relations

| Relation | Description |
|----------|-------------|
| `self` | Link to this object |
| `root` | Link to root catalog |
| `parent` | Link to parent catalog/collection |
| `child` | Link to child catalog/collection |
| `item` | Link to an item |
| `collection` | Link to collection |
