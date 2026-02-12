# Classification Methods

Implement supervised learning for land cover mapping using XEE, Scikit-Learn, and GeoAI techniques.

## Overview

This example covers:

1. **Machine Learning Algorithms**: Random Forest and Gradient Boosting (XGBoost).
2. **Statistical Classifiers**: Minimum Distance and Maximum Likelihood concepts.
3. **Evaluation Metrics**: Confusion Matrix, Kappa, and OA.
4. **Explainable GeoAI (X-GeoAI)**: Using feature importance to interpret model decisions.

## Step 1: Data Preparation and Label Loading

```python
import ee
import xarray as xr
import xee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define ROI
roi = ee.Geometry.Point([77.1025, 28.7041]).buffer(5000).bounds()

# Sentinel-2 Composite
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-03-31') \
    .median().clip(roi)

ds = xr.open_dataset(s2, engine='ee', geometry=roi, scale=10).compute()
```

## Step 2: Sampling and Training Data

In a real scenario, you would load a shapefile. For this example, we generate dummy training points within the ROI.

```python
# Classes: 1: Water, 2: Forest, 3: Urban
# (In practice, use ee.FeatureCollection)
def get_training_data(dataset):
    # Flatten dataset for sampling
    df = dataset[['B2', 'B3', 'B4', 'B8', 'B11', 'B12']].to_dataframe().dropna()
    
    # Simple rule-based labeling for this demonstration (Synthetic Training)
    df['label'] = 3 # Default Urban
    df.loc[(df.B8 - df.B4) / (df.B8 + df.B4) > 0.5, 'label'] = 2 # Forest
    df.loc[(df.B3 - df.B8) / (df.B3 + ds.B8) > 0.1, 'label'] = 1 # Water
    
    return df.sample(2000)

train_df = get_training_data(ds)
X_train = train_df[['B2', 'B3', 'B4', 'B8', 'B11', 'B12']]
y_train = train_df['label']
```

## Step 3: Random Forest Classification

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the full xarray dataset
# We can use xr.apply_ufunc for efficient prediction
def predict_rf(b2, b3, b4, b8, b11, b12):
    # Flatten bands and stack
    input_data = np.stack([b2.ravel(), b3.ravel(), b4.ravel(), b8.ravel(), b11.ravel(), b12.ravel()], axis=-1)
    
    # Handle NaNs
    mask = ~np.isnan(input_data).any(axis=1)
    preds = np.full(b2.size, 0)
    preds[mask] = rf.predict(input_data[mask])
    
    return preds.reshape(b2.shape)

classification = xr.apply_ufunc(
    predict_rf, ds.B2, ds.B3, ds.B4, ds.B8, ds.B11, ds.B12,
    dask='parallelized', output_dtypes=[np.int32]
)

plt.figure(figsize=(10, 10))
classification.plot(cmap='viridis')
plt.title("Random Forest Classification")
plt.show()
```

## Step 4: Explainable GeoAI (Feature Importance)

Understanding which bands contribute most to the classification.

```python
importances = rf.feature_importances_
features = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("X-GeoAI: Feature Importance")
plt.show()
```

## Step 5: Accuracy Assessment

```python
# Assuming we have independent test data
test_df = get_training_data(ds) # Sample again for test
y_pred = rf.predict(test_df[['B2', 'B3', 'B4', 'B8', 'B11', 'B12']])

print(classification_report(test_df['label'], y_pred))

# Confusion Matrix
cm = confusion_matrix(test_df['label'], y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```

## Key Takeaways

!!! success "Summary"
    - **Ensemble Models**: Random Forest is robust for most RS tasks.
    - **X-GeoAI**: Visualizing feature importance helps build trust in "black-box" models.
    - **Sampling**: Stratified sampling is crucial for unbiased accuracy.

→ Next: [MCDM Methods](mcdm-methods.md)
