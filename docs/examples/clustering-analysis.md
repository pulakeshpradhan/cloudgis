# Unsupervised Clustering for Land Cover with XEE

Unsupervised classification using K-Means and other clustering algorithms on satellite imagery accessed through XEE.

## Overview

This example demonstrates:

- Multi-spectral feature extraction
- K-Means clustering
- DBSCAN for spatial clustering
- Hierarchical clustering
- Cluster validation metrics
- Optimal cluster number selection

## Step 1: Load Multi-Spectral Data

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define region
roi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])

# Load Sentinel-2 data
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2023-06-01', '2023-08-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Create composite
composite = collection.median().clip(roi)

# Load with XEE
ds = xr.open_dataset(
    composite,
    engine='ee',
    geometry=roi,
    scale=20,
    crs='EPSG:4326'
).compute()

print("Data loaded:", ds.dims)
```

## Step 2: Prepare Features

```python
# Calculate spectral indices
ndvi = (ds.B8 - ds.B4) / (ds.B8 + ds.B4)
ndwi = (ds.B3 - ds.B8) / (ds.B3 + ds.B8)
ndbi = (ds.B11 - ds.B8) / (ds.B11 + ds.B8)

# Stack all features
features = np.stack([
    ds.B2.values.flatten(),  # Blue
    ds.B3.values.flatten(),  # Green
    ds.B4.values.flatten(),  # Red
    ds.B8.values.flatten(),  # NIR
    ds.B11.values.flatten(), # SWIR1
    ndvi.values.flatten(),
    ndwi.values.flatten(),
    ndbi.values.flatten()
], axis=1)

# Remove NaN values
valid_mask = ~np.isnan(features).any(axis=1)
features_valid = features[valid_mask]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_valid)

print(f"Valid pixels: {len(features_valid):,}")
print(f"Features shape: {features_scaled.shape}")
```

## Step 3: Determine Optimal Number of Clusters

```python
# Elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, labels))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'o-')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'o-', color='green')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")
```

## Step 4: K-Means Clustering

```python
# Perform K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_valid = kmeans.fit_predict(features_scaled)

# Map back to image dimensions
height, width = ds.B2.shape
labels_full = np.full(height * width, -1)
labels_full[valid_mask] = labels_valid
labels_image = labels_full.reshape(height, width)

# Visualize clusters
plt.figure(figsize=(12, 10))
plt.imshow(labels_image, cmap='tab10')
plt.colorbar(label='Cluster ID')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.axis('off')
plt.tight_layout()
plt.show()
```

## Step 5: DBSCAN Clustering

```python
# DBSCAN for density-based clustering
# Use subset for faster processing
sample_size = min(50000, len(features_scaled))
sample_idx = np.random.choice(len(features_scaled), sample_size, replace=False)
features_sample = features_scaled[sample_idx]

dbscan = DBSCAN(eps=0.5, min_samples=50)
labels_dbscan = dbscan.fit_predict(features_sample)

n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"DBSCAN found {n_clusters} clusters")
print(f"Noise points: {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)")
```

## Step 6: Hierarchical Clustering

```python
# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Use sample for dendrogram
sample_small = features_scaled[np.random.choice(len(features_scaled), 1000, replace=False)]

# Compute linkage
linkage_matrix = linkage(sample_small, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Apply hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
labels_hierarchical = hierarchical.fit_predict(features_scaled)

# Map to image
labels_hier_full = np.full(height * width, -1)
labels_hier_full[valid_mask] = labels_hierarchical
labels_hier_image = labels_hier_full.reshape(height, width)
```

## Step 7: Compare Clustering Methods

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# K-Means
axes[0].imshow(labels_image, cmap='tab10')
axes[0].set_title('K-Means Clustering')
axes[0].axis('off')

# DBSCAN (on sample)
axes[1].scatter(features_sample[:, 0], features_sample[:, 1], 
                c=labels_dbscan, cmap='tab10', s=1, alpha=0.5)
axes[1].set_title('DBSCAN (Feature Space)')
axes[1].set_xlabel('Feature 1 (Blue)')
axes[1].set_ylabel('Feature 2 (Green)')

# Hierarchical
axes[2].imshow(labels_hier_image, cmap='tab10')
axes[2].set_title('Hierarchical Clustering')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Step 8: Cluster Validation

```python
# Calculate validation metrics
methods = {
    'K-Means': labels_valid,
    'Hierarchical': labels_hierarchical
}

metrics_results = []

for method_name, labels in methods.items():
    silhouette = silhouette_score(features_scaled, labels)
    calinski = calinski_harabasz_score(features_scaled, labels)
    davies = davies_bouldin_score(features_scaled, labels)
    
    metrics_results.append({
        'Method': method_name,
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski,
        'Davies-Bouldin': davies
    })
    
    print(f"\n{method_name}:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Calinski-Harabasz: {calinski:.1f}")
    print(f"  Davies-Bouldin: {davies:.3f}")

import pandas as pd
metrics_df = pd.DataFrame(metrics_results)
print("\n", metrics_df)
```

## Step 9: Cluster Characterization

```python
# Analyze cluster characteristics
cluster_stats = []

for cluster_id in range(optimal_k):
    cluster_mask = labels_valid == cluster_id
    cluster_features = features_valid[cluster_mask]
    
    stats = {
        'Cluster': cluster_id,
        'Size': cluster_mask.sum(),
        'Blue_mean': cluster_features[:, 0].mean(),
        'Green_mean': cluster_features[:, 1].mean(),
        'Red_mean': cluster_features[:, 2].mean(),
        'NIR_mean': cluster_features[:, 3].mean(),
        'NDVI_mean': cluster_features[:, 5].mean(),
        'NDWI_mean': cluster_features[:, 6].mean()
    }
    cluster_stats.append(stats)

stats_df = pd.DataFrame(cluster_stats)
print("\nCluster Statistics:")
print(stats_df)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

stats_df.plot(x='Cluster', y='NDVI_mean', kind='bar', ax=axes[0, 0], color='green')
axes[0, 0].set_title('Mean NDVI by Cluster')
axes[0, 0].set_ylabel('NDVI')

stats_df.plot(x='Cluster', y='NDWI_mean', kind='bar', ax=axes[0, 1], color='blue')
axes[0, 1].set_title('Mean NDWI by Cluster')
axes[0, 1].set_ylabel('NDWI')

stats_df.plot(x='Cluster', y='NIR_mean', kind='bar', ax=axes[1, 0], color='red')
axes[1, 0].set_title('Mean NIR by Cluster')
axes[1, 0].set_ylabel('NIR Reflectance')

stats_df.plot(x='Cluster', y='Size', kind='bar', ax=axes[1, 1], color='gray')
axes[1, 1].set_title('Cluster Size')
axes[1, 1].set_ylabel('Number of Pixels')

plt.tight_layout()
plt.show()
```

## Step 10: Export Results

```python
# Create result dataset
clustering_results = xr.Dataset({
    'kmeans_labels': (['y', 'x'], labels_image),
    'hierarchical_labels': (['y', 'x'], labels_hier_image)
}, coords={
    'y': ds.lat.values,
    'x': ds.lon.values
})

# Add CRS
clustering_results = clustering_results.rio.write_crs('EPSG:4326')

# Save results
clustering_results.to_netcdf('clustering_results.nc')
print("Results saved to clustering_results.nc")

# Export cluster map as GeoTIFF
labels_da = xr.DataArray(
    labels_image,
    dims=['y', 'x'],
    coords={'y': ds.lat.values, 'x': ds.lon.values}
)
labels_da = labels_da.rio.write_crs('EPSG:4326')
labels_da.rio.to_raster('kmeans_clusters.tif')

# Save statistics
stats_df.to_csv('cluster_statistics.csv', index=False)
metrics_df.to_csv('clustering_metrics.csv', index=False)

print("All results exported")
```

## Key Takeaways

!!! success "What You Learned"
    - Multi-spectral feature extraction from XEE data
    - Optimal cluster number selection (Elbow & Silhouette)
    - K-Means clustering implementation
    - DBSCAN for density-based clustering
    - Hierarchical clustering with dendrograms
    - Cluster validation metrics
    - Cluster characterization and interpretation
    - Comparison of clustering methods

## Additional Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Cluster Validation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
