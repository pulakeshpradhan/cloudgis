# Deep Learning Architectures in Spatial Analysis

Leverage Convolutional Neural Networks (CNNs) and Transformers for advanced spatial feature extraction using XEE, TensorFlow, and PyTorch.

## Overview

This example covers:

1. **Convolutional Neural Networks (CNN)**: Implementing a U-Net for semantic segmentation.
2. **Vision Transformers (ViT)**: Concept of global attention in remote sensing.
3. **Graph Neural Networks (GNN)**: Analyzing irregularly spaced spatial data.
4. **Self-Supervised Learning**: Pre-training on massive unlabelled satellite imagery.

## Step 1: Data Preparation for Deep Learning

Deep learning requires smaller "patches" or "tiles" rather than massive full-scene images.

```python
import ee
import xarray as xr
import xee
import numpy as np

# Load Sentinel-2
roi = ee.Geometry.Point([77.1, 28.7]).buffer(1000).bounds()
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).median().clip(roi)

ds = xr.open_dataset(s2, engine='ee', geometry=roi, scale=10).compute()

# Stack bands and create patches
def make_patches(da, size=64):
    # (Simplified patch creation logic)
    data = da[['B2', 'B3', 'B4', 'B8']].to_array().values
    c, h, w = data.shape
    patch = data[:, :size, :size]
    return np.expand_dims(np.moveaxis(patch, 0, -1), 0) # (Batch, H, W, C)

patch_data = make_patches(ds)
print(f"Input shape: {patch_data.shape}")
```

## Step 2: CNN Architecture (U-Net in TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(64, 64, 4)):
    inputs = layers.Input(input_shape)
    
    # Downsample
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Bottleneck
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    
    # Upsample
    u3 = layers.UpSampling2D((2, 2))(c2)
    u3 = layers.concatenate([u3, c1])
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u3)
    
    return models.Model(inputs, outputs)

model = build_unet()
model.summary()
```

## Step 3: Transformers in Vision (Terminology)

Unlike CNNs, Transformers use **Self-Attention** to capture long-range dependencies in satellite imagery. This is particularly useful for detecting large-scale land forms or complex urban patterns.

```python
# Conceptual Vision Transformer block
class AttentionBlock(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=8, key_dim=embed_dim)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        attn_out = self.mha(x, x)
        return self.norm(x + attn_out)
```

## Step 4: Graph Neural Networks (GNN)

Used when data is not a grid (e.g., weather stations, social sensing data, or object-based image analysis).

```python
# Conceptual GNN Layer (PyTorch Geometric style)
# Each node (pixel/object) aggregates information from its spatial neighbors.
# x_new = f(x, neighbors)
```

## Step 5: Training and Evaluation

```python
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_gen, epochs=10)
```

## Key Takeaways

!!! success "Summary"
    - **CNNs**: The standard for pixel-wise classification and object detection.
    - **Transformers**: Rising popularity for large-scale "Foundation Models".
    - **Infrastructure**: Processing DL models requires GPU; it's often best to export XEE data to TFRecord or Zarr for training.

→ Back to [Index](../index.md)
