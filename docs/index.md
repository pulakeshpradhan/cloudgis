# Cloud Native Remote Sensing with Python

Welcome to the comprehensive guide on cloud-native remote sensing with Python! This course covers modern tools and techniques for working with large-scale Earth observation datasets using cloud-based approaches.

## 🎯 What You'll Learn

This course provides a structured introduction to the essential Python libraries and cloud-native technologies for remote sensing:

- **XArray**: Multi-dimensional labeled arrays for efficient data manipulation
- **STAC (Spatio Temporal Asset Catalog)**: Standardized way to discover and access geospatial data
- **Dask**: Parallel computing library for scaling your analysis
- **Zarr**: Cloud-optimized array storage format
- **XEE**: XArray Earth Engine integration for seamless data access

## 🚀 Why Cloud-Native Remote Sensing?

Traditional remote sensing workflows often involve downloading large datasets to local machines, which is:

- **Time-consuming**: Downloading terabytes of data takes hours or days
- **Storage-intensive**: Requires significant local storage capacity
- **Computationally limited**: Constrained by local machine resources
- **Not scalable**: Difficult to process large areas or long time series

Cloud-native approaches solve these problems by:

✅ **Accessing data on-demand** - Only download what you need  
✅ **Leveraging cloud compute** - Scale processing power as needed  
✅ **Using optimized formats** - Zarr and COG for efficient data access  
✅ **Parallel processing** - Dask for distributed computing  
✅ **Standardized discovery** - STAC for finding relevant datasets  

## 📚 Course Structure

### Getting Started

Learn the basics of setting up your environment and working with Google Colab for cloud-based analysis.

### Fundamentals

Master the core libraries: XArray for data manipulation, STAC for data discovery, Dask for parallel computing, Zarr for storage, and XEE for Earth Engine integration.

### Data Processing

Apply your knowledge to real-world tasks like calculating spectral indices, masking clouds, extracting time series, and aggregating data.

### Advanced Topics

Scale your analysis with advanced Dask techniques, cloud computing platforms, and optimization strategies.

### Practical Examples

Work through complete examples including NDVI analysis, land cover classification, change detection, and multi-temporal analysis.

## 🛠️ Key Technologies

### XArray

XArray extends NumPy and Pandas to N-dimensional labeled arrays, making it perfect for satellite imagery with dimensions like time, latitude, longitude, and bands.

```python
import xarray as xr

# Open a dataset
ds = xr.open_dataset('sentinel2_data.nc')

# Select data by labels
subset = ds.sel(time='2023-01-01', band='red')

# Perform calculations
ndvi = (ds['nir'] - ds['red']) / (ds['nir'] + ds['red'])
```

### STAC

STAC provides a common language to describe geospatial information, making it easier to discover and access satellite imagery.

```python
from pystac_client import Client

# Connect to a STAC catalog
catalog = Client.open('https://earth-search.aws.element84.com/v1')

# Search for Sentinel-2 imagery
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[lon_min, lat_min, lon_max, lat_max],
    datetime='2023-01-01/2023-12-31'
)
```

### Dask

Dask enables parallel computing in Python, allowing you to work with datasets larger than memory.

```python
import dask.array as da

# Create a Dask array
x = da.from_zarr('large_dataset.zarr')

# Computations are lazy
result = x.mean(axis=0)

# Trigger computation
result.compute()
```

### Zarr

Zarr is a cloud-optimized format for storing chunked, compressed N-dimensional arrays.

```python
import zarr

# Create a Zarr array
z = zarr.open('data.zarr', mode='w', shape=(10000, 10000), 
              chunks=(1000, 1000), dtype='f4')

# Write data
z[:] = data

# Read data
subset = z[1000:2000, 1000:2000]
```

### XEE

XEE (XArray Earth Engine Extension) allows you to use Earth Engine datasets with XArray.

```python
import xee
import ee

ee.Initialize(project='spatialgeography')

# Open an Earth Engine ImageCollection as XArray
ds = xr.open_dataset(
    'ee://COPERNICUS/S2_SR',
    engine='ee',
    geometry=ee.Geometry.Point([lon, lat]).buffer(10000),
    scale=10
)
```

## 💡 Prerequisites

- **Python Programming**: Basic knowledge of Python
- **NumPy/Pandas**: Familiarity with array operations
- **Remote Sensing**: Basic understanding of satellite imagery
- **GIS Concepts**: Knowledge of coordinate systems and projections

## 🎓 Learning Path

1. **Beginners**: Start with Getting Started → Fundamentals
2. **Intermediate**: Focus on Data Processing → Practical Examples
3. **Advanced**: Dive into Advanced Topics and optimization

## 🌟 Key Features

- **Hands-on Examples**: Every concept includes working code examples
- **Google Colab Integration**: Run examples in your browser without setup
- **Real-world Datasets**: Work with actual Sentinel-2 and other satellite data
- **Best Practices**: Learn industry-standard approaches
- **Scalable Solutions**: Techniques that work from local to cloud scale

## 📖 How to Use This Course

1. **Read sequentially** for a complete learning experience
2. **Jump to specific topics** using the navigation menu
3. **Run the code examples** in Google Colab or your local environment
4. **Complete the exercises** to reinforce your learning
5. **Explore the reference** section for detailed API information

## 🤝 Contributing

This course is open-source and welcomes contributions! If you find errors, have suggestions, or want to add content, please contribute via GitHub.

## 📝 License

This course content is available under the Creative Commons Attribution 4.0 International License.

---

Ready to get started? Head to [Introduction](getting-started/introduction.md) to begin your cloud-native remote sensing journey!

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Set up your environment and learn the basics of cloud-native remote sensing workflows.

    [:octicons-arrow-right-24: Start Learning](getting-started/introduction.md)

- :material-book-open-variant:{ .lg .middle } **Fundamentals**

    ---

    Master XArray, STAC, Dask, Zarr, and XEE - the core tools for cloud-native analysis.

    [:octicons-arrow-right-24: Learn Fundamentals](fundamentals/xarray-basics.md)

- :material-cog:{ .lg .middle } **Data Processing**

    ---

    Apply techniques for spectral indices, cloud masking, time series, and aggregation.

    [:octicons-arrow-right-24: Process Data](processing/spectral-indices.md)

- :material-rocket:{ .lg .middle } **Advanced Topics**

    ---

    Scale your analysis with Dask, cloud platforms, and optimization strategies.

    [:octicons-arrow-right-24: Go Advanced](advanced/scaling-dask.md)

- :material-code-braces:{ .lg .middle } **Practical Examples**

    ---

    Complete workflows for NDVI analysis, land cover, change detection, and more.

    [:octicons-arrow-right-24: See Examples](examples/ndvi-analysis.md)

- :material-book-multiple:{ .lg .middle } **Reference**

    ---

    Comprehensive API documentation and best practices for all libraries.

    [:octicons-arrow-right-24: Browse Reference](reference/xarray-api.md)

</div>
