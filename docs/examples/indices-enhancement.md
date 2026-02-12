# Remote Sensing Indices and Enhancement

A comprehensive guide to calculating spectral indices and applying enhancement techniques using XEE, XArray, and Dask.

## Overview

This example covers:

- Calculating a broad range of spectral indices (Vegetation, Water, Urban, Burn).
- Image enhancement techniques (Contrast stretching, Histogram equalization).
- Spatial filtering and edge detection.
- Sensor-specific considerations (Sentinel-2 vs. Landsat).

## Step 1: Initialization and Data Loading

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
import geemap

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define a Region of Interest (ROI)
roi = ee.Geometry.Point([77.1025, 28.7041]).buffer(5000).bounds()

# Load Sentinel-2 SR data
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .median() \
    .clip(roi)

# Load with XEE
ds = xr.open_dataset(s2, engine='ee', geometry=roi, scale=10)
ds = ds.compute() # Bring into memory for local enhancement
```

## Step 2: Comprehensive Spectral Indices with Spyndex

While manual calculations work, using the [spyndex](https://github.com/awesome-spectral-indices/spyndex) library is the scientifically standard approach as it leverages the [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices) database.

```python
import spyndex

# Calculate multiple indices at once
# spyndex automatically handles the constants and formulas
indices = spyndex.computeIndex(
    index = ["NDVI", "EVI", "SAVI", "NDWI", "MNDWI", "NDBI", "NBR"],
    params = {
        "N": ds.B8,   # NIR
        "R": ds.B4,   # Red
        "G": ds.B3,   # Green
        "B": ds.B2,   # Blue
        "S1": ds.B11, # SWIR1
        "S2": ds.B12, # SWIR2
        "L": 0.5      # Soil adjustment factor for SAVI
    }
)

# Merge back into our main dataset
ds = xr.merge([ds, indices])

print("Indices calculated via spyndex.")
```

## Step 3: Image Enhancement Techniques

### 3.1 Contrast Stretching (2% Linear Stretch)

```python
def linear_stretch(array, percent=2):
    low, high = np.nanpercentile(array, [percent, 100-percent])
    stretched = (array - low) / (high - low)
    return np.clip(stretched, 0, 1)

# Apply to RGB bands
r = linear_stretch(ds.B4.values)
g = linear_stretch(ds.B3.values)
b = linear_stretch(ds.B2.values)

rgb_stretched = np.stack([r, g, b], axis=-1)

plt.figure(figsize=(10, 10))
plt.imshow(rgb_stretched)
plt.title("Enhanced True Color (2% Linear Stretch)")
plt.axis('off')
plt.show()
```

### 3.2 Histogram Equalization

```python
from skimage import exposure

# Apply histogram equalization to the Green band as an example
g_eq = exposure.equalize_hist(ds.B3.values)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(ds.B3, cmap='gray'), plt.title('Original Green')
plt.subplot(122), plt.imshow(g_eq, cmap='gray'), plt.title('Equalized Green')
plt.show()
```

## Step 4: Spatial Operations & Filtering

Using `scipy.ndimage` or `skimage` for spatial enhancement.

```python
from scipy import ndimage

# Sobel Edge Detection on NDVI
ndvi_filled = ds.NDVI.fillna(0).values
sobel_x = ndimage.sobel(ndvi_filled, axis=0)
sobel_y = ndimage.sobel(ndvi_filled, axis=1)
sobel_mag = np.hypot(sobel_x, sobel_y)

plt.figure(figsize=(8, 8))
plt.imshow(sobel_mag, cmap='magma')
plt.title("NDVI Edge Detection (Sobel)")
plt.axis('off')
plt.show()
```

## Step 5: Visualization with Geemap

```python
Map = geemap.Map()
Map.centerObject(roi, 13)

vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
Map.addLayer(s2, vis_params, 'Sentinel-2 RGB')

# Add NDVI from our XArray computation back (conceptual)
# In geemap, you can directly calculate or use the EE object
ndvi_ee = s2.normalizedDifference(['B8', 'B4'])
Map.addLayer(ndvi_ee, {'min': 0, 'max': 1, 'palette': ['white', 'green']}, 'NDVI')

Map
```

## Key Takeaways

!!! success "Summary"
    - XEE provides the bridge to load raw EE data into XArray.
    - XArray allows for clear, vectorized index calculation.
    - Python’s scientific stack (NumPy, Scipy, Skimage) offers advanced enhancement tools not natively in EE.
    - Geemap remains the best tool for interactive validation.

→ Next: [Clustering Methods](clustering-methods.md)
