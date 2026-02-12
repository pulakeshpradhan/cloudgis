# Clustering Methods in Geospatial Analysis

Explore unsupervised learning techniques for grouping spatial data using XEE, Scikit-Learn, and Dask.

## Overview

This example covers:

1. **Statistical Clustering**: K-Means.
2. **Machine Learning Clustering**: DBSCAN and Gaussian Mixture Models (GMM).
3. **Deep Learning-derived Clustering**: Using Autoencoders for dimensionality reduction before clustering.
4. **Spatially Constrained Clustering**: Regionalization techniques.
5. **Evaluation Metrics**: Silhouette and Calinski-Harabasz.

## Step 1: Load Multi-Spectral Data

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Load Sentinel-2 median composite
roi = ee.Geometry.Rectangle([77.0, 28.5, 77.2, 28.7])
image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-03-31') \
    .median().clip(roi)

# Load into XArray
ds = xr.open_dataset(image, engine='ee', geometry=roi, scale=20)
ds = ds.compute()
```

## Step 2: Feature Engineering and Scaling

```python
# Create a feature stack
data = np.stack([
    ds.B2, ds.B3, ds.B4, ds.B8, ds.B11,
    (ds.B8 - ds.B4) / (ds.B8 + ds.B4) # NDVI
], axis=-1)

# Flatten for clustering
rows, cols, bands = data.shape
X = data.reshape(-1, bands)

# Remove NaNs
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
```

## Step 3: Traditional & Statistical Clustering (K-Means)

```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

# Reshape back to image
result_km = np.full((rows * cols), np.nan)
result_km[mask] = labels_km
result_km = result_km.reshape(rows, cols)

plt.figure(figsize=(8, 8))
plt.imshow(result_km, cmap='terrain')
plt.title('K-Means Clustering (k=5)')
plt.show()
```

## Step 4: Machine Learning Based Clustering (GMM)

Gaussian Mixture Models provide probabilistic cluster assignments.

```python
gmm = GaussianMixture(n_components=5, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)

result_gmm = np.full((rows * cols), np.nan)
result_gmm[mask] = labels_gmm
result_gmm = result_gmm.reshape(rows, cols)

plt.figure(figsize=(8, 8))
plt.imshow(result_gmm, cmap='terrain')
plt.title('Gaussian Mixture Model')
plt.show()
```

## Step 5: Spatially Constrained Clustering (SKATER)

In geospatial analysis, we often want to ensure that clusters are spatially contiguous. The [SKATER](https://pysal.org/spopt/notebooks/skater.html) (Spatial K'luster Analysis by Tree Edge Removal) algorithm from the `spopt` library is the scientific standard for this.

```python
from libpysal.weights import lat2W
from spopt.region import Skater

# 1. Create a spatial weights matrix (contiguity)
w = lat2W(rows, cols)

# 2. Initialize and fit SKATER
# n_clusters=8, floor=None
model = Skater(X_scaled, w, n_clusters=8)
model.solve()

# 3. Reshape labels back to image
result_skater = np.full((rows * cols), np.nan)
result_skater[mask] = model.labels_
result_skater = result_skater.reshape(rows, cols)

plt.figure(figsize=(8, 8))
plt.imshow(result_skater, cmap='tab20')
plt.title('SKATER Spatially Constrained Clustering')
plt.show()
```

## Step 6: Evaluation Metrics

```python
# Silhouette Score (Computationally intensive, use sample)
sample_idx = np.random.choice(len(X_scaled), 5000)
score = silhouette_score(X_scaled[sample_idx], labels_km[sample_idx])
print(f"Silhouette Score (K-Means): {score:.3f}")
```

## Key Takeaways

!!! success "Summary"
    - **Feature Selection**: Adding NDVI or texture improves cluster separation.
    - **Scaling**: Essential for distance-based methods like K-Means.
    - **Spatial Constraints**: Helps in creating regions rather than just spectral groupings.
    - **Evaluation**: Use scores to determine the "elbow" for optimal cluster count.

→ Next: [Classification Methods](classification-methods.md)
