# Multi-Criteria Decision Making (MCDM)

Perform site suitability analysis using Ranking, Weighting, and Aggregation methods with XEE and XArray.

## Overview

This example covers:

1. **Criteria Selection**: Slope, LULC, Distance to Roads, etc.
2. **Normalization Methods**: Preparing diverse layers for comparison.
3. **Weighting Methods**: Simple Additive Weighting (SAW) and TOPSIS logic.
4. **Ranking and Aggregation**: Generating a final suitability map.

## Step 1: Initialize Criteria Layers

We'll load topography and land cover data to find the best site for a nature park.

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Area: Near a mountain region
roi = ee.Geometry.Rectangle([76.5, 31.0, 77.0, 31.5])

# Layer 1: Slope (Derived from SRTM)
dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
slope = ee.Terrain.slope(dem)

# Layer 2: Distance to Water (Derived from JRC)
water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence').clip(roi)
dist_water = water.distance(ee.Kernel.gaussian(5000, 3000, 'meters'))

# Load into XArray
ds = xr.open_dataset(ee.Image.cat([slope.rename('slope'), dist_water.rename('dist_water')]), 
                    engine='ee', geometry=roi, scale=100).compute()
```

## Step 2: Normalization Methods

MCDM requires criteria to be on the same scale (usually 0 to 1).

```python
def normalize_benefit(da):
    """Higher is better."""
    return (da - da.min()) / (da.max() - da.min())

def normalize_cost(da):
    """Lower is better (e.g., Slope)."""
    return (da.max() - da) / (da.max() - da.min())

# Slope: Lower is better for construction (Cost)
norm_slope = normalize_cost(ds.slope)

# Distance to Water: Closer is better (Cost)
norm_water = normalize_cost(ds.dist_water)
```

## Step 3: Weighted Aggregation (SAW Method)

The Simple Additive Weighting (SAW) is the most intuitive aggregation method.

```python
# Weights: 40% Slope, 60% Water proximity
w_slope = 0.4
w_water = 0.6

suitability_saw = (norm_slope * w_slope) + (norm_water * w_water)

plt.figure(figsize=(10, 8))
suitability_saw.plot(cmap='YlGn')
plt.title("Suitability Map (SAW Method)")
plt.show()
```

## Step 4: Rigorous Ranking with PyMCDM (TOPSIS)

For a scientifically standard approach, we use the `pymcdm` library to calculate the TOPSIS score.

```python
from pymcdm import methods as mcdm_methods
from pymcdm import helpers

# 1. Prepare data matrix (alternatives x criteria)
# Every pixel is an alternative
X = np.stack([norm_slope.values.ravel(), norm_water.values.ravel()], axis=1)

# 2. Define weights and criteria types (1 for benefit, -1 for cost)
# Since we already normalized, they can both be treated as benefit (higher score is better)
weights = np.array([0.4, 0.6])
types = np.array([1, 1])

# 3. Initialize TOPSIS method
topsis = mcdm_methods.TOPSIS()

# 4. Calculate preferences
# Handle large datasets by processing in blocks if necessary
pref = topsis(X, weights, types)

# 5. Reshape back to spatial dimensions
topsis_score = pref.reshape(norm_slope.shape)

plt.figure(figsize=(10, 8))
plt.imshow(topsis_score, cmap='RdYlGn')
plt.colorbar(label='Preference Score')
plt.title("Suitability Map (PyMCDM TOPSIS)")
plt.show()
```

## Key Takeaways

!!! success "Summary"
    - **Normalization**: Min-Max normalization is the foundation of spatial MCDM.
    - **Methods**: SAW is easy to implement; TOPSIS is more robust against outliers.
    - **Spatial Logic**: Every pixel acts as an "alternative" in the decision matrix.

→ Next: [Time Series and Phenology](timeseries-phenology.md)
