# Land Cover Classification with XEE

This example demonstrates how to perform supervised land cover classification using Earth Engine data accessed through XEE.

## Overview

We'll classify land cover into categories:

- Water
- Vegetation
- Urban/Built-up
- Bare Soil

**Tools**: XEE, XArray, Scikit-learn, Matplotlib

## Step 1: Initialize and Load Data

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define region of interest (example: area around a city)
roi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])  # Delhi region

# Load Sentinel-2 data
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-03-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

print(f"Found {collection.size().getInfo()} images")
```

## Step 2: Create Cloud-Free Composite

```python
def mask_s2_clouds(image):
    """Mask clouds using QA60 band."""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                 qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Apply cloud mask and create median composite
composite = collection.map(mask_s2_clouds).median().clip(roi)

# Open with XEE
ds = xr.open_dataset(
    composite,
    engine='ee',
    geometry=roi,
    scale=20,  # 20m resolution
    crs='EPSG:4326'
)

print(ds)
```

## Step 3: Calculate Spectral Indices

```python
# Calculate indices
ndvi = (ds.B8 - ds.B4) / (ds.B8 + ds.B4)
ndwi = (ds.B3 - ds.B8) / (ds.B3 + ds.B8)
ndbi = (ds.B11 - ds.B8) / (ds.B11 + ds.B8)

# Add to dataset
ds['NDVI'] = ndvi
ds['NDWI'] = ndwi
ds['NDBI'] = ndbi

# Compute to load into memory
ds = ds.compute()
```

## Step 4: Prepare Training Data

```python
# Define training samples (you would normally collect these from field data)
# Format: [lon, lat, class_id]
# Classes: 0=Water, 1=Vegetation, 2=Urban, 3=Bare Soil

training_points = [
    # Water samples
    [77.15, 28.65, 0], [77.18, 28.62, 0], [77.20, 28.68, 0],
    # Vegetation samples
    [77.10, 28.55, 1], [77.25, 28.70, 1], [77.30, 28.60, 1],
    # Urban samples
    [77.20, 28.60, 2], [77.22, 28.58, 2], [77.18, 28.56, 2],
    # Bare soil samples
    [77.12, 28.48, 3], [77.28, 28.52, 3], [77.32, 28.54, 3]
]

# Extract features at training points
X_train = []
y_train = []

for lon, lat, class_id in training_points:
    # Extract all features at this point
    point_data = ds.sel(lon=lon, lat=lat, method='nearest')
    
    features = [
        point_data.B2.values.item(),  # Blue
        point_data.B3.values.item(),  # Green
        point_data.B4.values.item(),  # Red
        point_data.B8.values.item(),  # NIR
        point_data.B11.values.item(), # SWIR1
        point_data.NDVI.values.item(),
        point_data.NDWI.values.item(),
        point_data.NDBI.values.item()
    ]
    
    X_train.append(features)
    y_train.append(class_id)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training samples: {len(y_train)}")
```

## Step 5: Train Classifier

```python
# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Feature importance
feature_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'NDVI', 'NDWI', 'NDBI']
importance = clf.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance)
plt.xlabel('Importance')
plt.title('Feature Importance for Land Cover Classification')
plt.tight_layout()
plt.show()
```

## Step 6: Classify Entire Image

```python
# Prepare all pixels for classification
height, width = ds.B2.shape
n_pixels = height * width

# Stack all features
features_stack = np.stack([
    ds.B2.values.flatten(),
    ds.B3.values.flatten(),
    ds.B4.values.flatten(),
    ds.B8.values.flatten(),
    ds.B11.values.flatten(),
    ds.NDVI.values.flatten(),
    ds.NDWI.values.flatten(),
    ds.NDBI.values.flatten()
], axis=1)

# Handle NaN values
valid_mask = ~np.isnan(features_stack).any(axis=1)
features_valid = features_stack[valid_mask]

# Classify
predictions = np.full(n_pixels, -1)  # -1 for no data
predictions[valid_mask] = clf.predict(features_valid)

# Reshape to image
classification = predictions.reshape(height, width)
```

## Step 7: Visualize Results

```python
# Define colors for each class
colors = ['blue', 'green', 'red', 'yellow']
class_names = ['Water', 'Vegetation', 'Urban', 'Bare Soil']

# Create custom colormap
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors)

# Plot classification result
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# True color composite
rgb = np.stack([
    ds.B4.values / 3000,  # Red
    ds.B3.values / 3000,  # Green
    ds.B2.values / 3000   # Blue
], axis=-1)
rgb = np.clip(rgb, 0, 1)

axes[0].imshow(rgb)
axes[0].set_title('True Color Composite')
axes[0].axis('off')

# Classification
im = axes[1].imshow(classification, cmap=cmap, vmin=0, vmax=3)
axes[1].set_title('Land Cover Classification')
axes[1].axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                   for i in range(len(class_names))]
axes[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
```

## Step 8: Calculate Class Statistics

```python
# Calculate area for each class
pixel_area = 20 * 20  # 20m x 20m pixels in square meters
areas_sqkm = {}

for class_id, class_name in enumerate(class_names):
    pixel_count = np.sum(classification == class_id)
    area_sqkm = (pixel_count * pixel_area) / 1e6
    areas_sqkm[class_name] = area_sqkm
    print(f"{class_name}: {area_sqkm:.2f} km²")

# Pie chart of land cover distribution
plt.figure(figsize=(8, 8))
plt.pie(areas_sqkm.values(), labels=areas_sqkm.keys(), 
        colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Land Cover Distribution')
plt.axis('equal')
plt.show()
```

## Step 9: Export Results

```python
# Create XArray DataArray for classification
classification_da = xr.DataArray(
    classification,
    dims=['y', 'x'],
    coords={
        'y': ds.lat.values,
        'x': ds.lon.values
    },
    name='land_cover'
)

# Add CRS information
classification_da = classification_da.rio.write_crs('EPSG:4326')

# Save as GeoTIFF
classification_da.rio.to_raster('land_cover_classification.tif')
print("Classification saved to land_cover_classification.tif")

# Save statistics
import pandas as pd
stats_df = pd.DataFrame({
    'Class': class_names,
    'Area_km2': list(areas_sqkm.values())
})
stats_df.to_csv('land_cover_stats.csv', index=False)
print("Statistics saved to land_cover_stats.csv")
```

## Advanced: Accuracy Assessment

```python
# If you have validation points (separate from training)
validation_points = [
    [77.16, 28.64, 0],  # Water
    [77.11, 28.56, 1],  # Vegetation
    [77.21, 28.59, 2],  # Urban
    [77.13, 28.49, 3]   # Bare soil
]

# Extract predictions at validation points
y_true = []
y_pred = []

for lon, lat, true_class in validation_points:
    # Find nearest pixel
    lat_idx = np.argmin(np.abs(ds.lat.values - lat))
    lon_idx = np.argmin(np.abs(ds.lon.values - lon))
    
    pred_class = classification[lat_idx, lon_idx]
    
    y_true.append(true_class)
    y_pred.append(pred_class)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
```

## Key Takeaways

!!! success "What You Learned"
    - Loading Earth Engine data with XEE for classification
    - Creating cloud-free composites
    - Calculating multiple spectral indices as features
    - Training Random Forest classifier
    - Classifying entire images
    - Visualizing classification results
    - Calculating area statistics
    - Exporting results as GeoTIFF

## Next Steps

→ Continue to [Change Detection](change-detection.md)

## Additional Resources

- [Scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html)
- [Earth Engine Land Cover](https://developers.google.com/earth-engine/tutorials/tutorial_api_06)
