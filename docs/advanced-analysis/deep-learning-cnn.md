# Deep Learning for Land Cover with CNNs

Convolutional Neural Networks (CNNs) for semantic segmentation of satellite imagery using XEE data.

## Overview

This example demonstrates:

- Loading multi-temporal Sentinel-2 data via XEE
- Preparing training data for CNNs
- Building U-Net architecture for segmentation
- Training with TensorFlow/Keras
- Prediction and accuracy assessment
- Comparison with Random Forest

## Step 1: Setup and Data Loading

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define study area
roi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])

# Load Sentinel-2 collection
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Create composite
composite = collection.median().clip(roi)

# Load with XEE
ds = xr.open_dataset(
    composite,
    engine='ee',
    geometry=roi,
    scale=10,  # 10m resolution
    crs='EPSG:4326'
).compute()

print(f"Data shape: {ds.B4.shape}")
```

## Step 2: Prepare Training Data

```python
# Stack bands for CNN input
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
image_stack = np.stack([ds[band].values for band in bands], axis=-1)

# Normalize to 0-1
image_normalized = image_stack / 10000.0
image_normalized = np.clip(image_normalized, 0, 1)

print(f"Image shape: {image_normalized.shape}")
print(f"Value range: [{image_normalized.min():.3f}, {image_normalized.max():.3f}]")

# Create synthetic labels (in practice, use actual ground truth)
# Classes: 0=Background, 1=Water, 2=Vegetation, 3=Urban, 4=Bare
height, width, channels = image_normalized.shape

# Simple rule-based labels for demonstration
ndvi = (ds.B8 - ds.B4) / (ds.B8 + ds.B4)
ndwi = (ds.B3 - ds.B8) / (ds.B3 + ds.B8)
ndbi = (ds.B11 - ds.B8) / (ds.B11 + ds.B8)

labels = np.zeros((height, width), dtype=np.int32)
labels[ndwi.values > 0.3] = 1  # Water
labels[(ndvi.values > 0.4) & (ndwi.values <= 0.3)] = 2  # Vegetation
labels[(ndbi.values > 0.1) & (ndvi.values <= 0.4)] = 3  # Urban
labels[(ndvi.values <= 0.2) & (ndbi.values <= 0.1)] = 4  # Bare

print(f"Labels shape: {labels.shape}")
print(f"Unique classes: {np.unique(labels)}")
```

## Step 3: Create Image Patches

```python
def create_patches(image, labels, patch_size=64, stride=32):
    """Create overlapping patches from image and labels."""
    patches_img = []
    patches_lbl = []
    
    h, w, c = image.shape
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch_img = image[i:i+patch_size, j:j+patch_size, :]
            patch_lbl = labels[i:i+patch_size, j:j+patch_size]
            
            # Skip patches with too many NaN values
            if np.isnan(patch_img).sum() < (patch_size * patch_size * 0.1):
                patches_img.append(patch_img)
                patches_lbl.append(patch_lbl)
    
    return np.array(patches_img), np.array(patches_lbl)

# Create patches
patch_size = 64
X_patches, y_patches = create_patches(image_normalized, labels, patch_size=patch_size)

print(f"Number of patches: {len(X_patches)}")
print(f"Patch shape: {X_patches[0].shape}")

# Replace NaN with 0
X_patches = np.nan_to_num(X_patches, 0)

# Convert labels to one-hot encoding
num_classes = 5
y_patches_onehot = tf.keras.utils.to_categorical(y_patches, num_classes)

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_patches, y_patches_onehot, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
```

## Step 4: Build U-Net Model

```python
def build_unet(input_shape, num_classes):
    """Build U-Net architecture for semantic segmentation."""
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder (upsampling)
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Build model
input_shape = (patch_size, patch_size, len(bands))
model = build_unet(input_shape, num_classes)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]
)

model.summary()
```

## Step 5: Train the Model

```python
# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 6: Prediction on Full Image

```python
def predict_full_image(model, image, patch_size=64, stride=32):
    """Predict on full image using sliding window."""
    h, w, c = image.shape
    prediction = np.zeros((h, w, num_classes))
    count = np.zeros((h, w))
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patch = np.nan_to_num(patch, 0)
            patch = np.expand_dims(patch, 0)
            
            pred = model.predict(patch, verbose=0)[0]
            prediction[i:i+patch_size, j:j+patch_size, :] += pred
            count[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping predictions
    count = np.maximum(count, 1)
    prediction = prediction / count[:, :, np.newaxis]
    
    return np.argmax(prediction, axis=-1)

# Predict
print("Predicting on full image...")
prediction_cnn = predict_full_image(model, image_normalized, patch_size, stride=32)

print(f"Prediction shape: {prediction_cnn.shape}")
print(f"Predicted classes: {np.unique(prediction_cnn)}")
```

## Step 7: Visualize Results

```python
# Define colors and labels
colors = ['black', 'blue', 'green', 'red', 'yellow']
class_names = ['Background', 'Water', 'Vegetation', 'Urban', 'Bare Soil']

from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RGB composite
rgb = np.stack([
    ds.B4.values / 3000,
    ds.B3.values / 3000,
    ds.B2.values / 3000
], axis=-1)
rgb = np.clip(rgb, 0, 1)

axes[0].imshow(rgb)
axes[0].set_title('True Color Composite')
axes[0].axis('off')

# Ground truth (synthetic)
im1 = axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=4)
axes[1].set_title('Ground Truth (Synthetic)')
axes[1].axis('off')

# CNN Prediction
im2 = axes[2].imshow(prediction_cnn, cmap=cmap, vmin=0, vmax=4)
axes[2].set_title('CNN Prediction (U-Net)')
axes[2].axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                   for i in range(len(class_names))]
fig.legend(handles=legend_elements, loc='lower center', ncol=5)

plt.tight_layout()
plt.show()
```

## Step 8: Accuracy Assessment

```python
# Flatten for metrics
y_true = labels.flatten()
y_pred = prediction_cnn.flatten()

# Remove background class
mask = y_true > 0
y_true_masked = y_true[mask]
y_pred_masked = y_pred[mask]

# Confusion matrix
cm = confusion_matrix(y_true_masked, y_pred_masked)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[1:], yticklabels=class_names[1:])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - CNN Classification')
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_masked, y_pred_masked, 
                           target_names=class_names[1:]))

# Overall accuracy
accuracy = np.mean(y_true_masked == y_pred_masked)
print(f"\nOverall Accuracy: {accuracy:.3f}")
```

## Step 9: Compare with Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Prepare data for RF
X_rf = image_normalized.reshape(-1, len(bands))
y_rf = labels.flatten()

# Remove NaN and background
valid_mask = ~np.isnan(X_rf).any(axis=1) & (y_rf > 0)
X_rf_valid = X_rf[valid_mask]
y_rf_valid = y_rf[valid_mask]

# Train RF
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_rf_valid, y_rf_valid)

# Predict
X_rf_all = np.nan_to_num(X_rf, 0)
prediction_rf = rf.predict(X_rf_all).reshape(height, width)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(prediction_rf, cmap=cmap, vmin=0, vmax=4)
axes[0].set_title('Random Forest Prediction')
axes[0].axis('off')

axes[1].imshow(prediction_cnn, cmap=cmap, vmin=0, vmax=4)
axes[1].set_title('CNN (U-Net) Prediction')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Accuracy comparison
rf_accuracy = np.mean(y_rf_valid == rf.predict(X_rf_valid))
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print(f"CNN Accuracy: {accuracy:.3f}")
```

## Step 10: Save Model and Results

```python
# Save model
model.save('unet_land_cover.h5')
print("Model saved to unet_land_cover.h5")

# Save predictions as GeoTIFF
prediction_da = xr.DataArray(
    prediction_cnn,
    dims=['y', 'x'],
    coords={'y': ds.lat.values, 'x': ds.lon.values}
)
prediction_da = prediction_da.rio.write_crs('EPSG:4326')
prediction_da.rio.to_raster('cnn_land_cover.tif')

print("Prediction saved to cnn_land_cover.tif")
```

## Key Takeaways

!!! success "What You Learned"
    - Loading satellite imagery for deep learning with XEE
    - Preparing training data (patches, normalization)
    - Building U-Net architecture for segmentation
    - Training CNNs with TensorFlow/Keras
    - Sliding window prediction on large images
    - Accuracy assessment for semantic segmentation
    - Comparison with traditional ML (Random Forest)
    - Model saving and deployment

## Additional Resources

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [TensorFlow Segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
- [Satellite Image Segmentation](https://github.com/topics/satellite-image-segmentation)
