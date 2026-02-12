# Vegetation Continuous Fields (VCF) Analysis

Analyze long-term vegetation trends using MODIS MOD44B data, XEE, and XArray visualization tools.

## Overview

This example demonstrates how to:

1. **Load Yearly Data**: Access the MODIS Vegetation Continuous Fields (VCF) product.
2. **Spatial Mapping**: Create maps of percent tree cover.
3. **Time Series Charts**: Generate line graphs of vegetation trends over decades.
4. **Statistical Distribution**: Analyze the distribution of vegetation classes using histograms.

## Step 1: Open the Dataset with XEE

We'll load the MOD44B collection, which provides yearly estimates of tree cover, non-tree vegetation, and non-vegetated surfaces.

```python
import xarray as xr
import xee
import ee
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define a Region of Interest (ROI) - a forested area or transition zone
roi = ee.Geometry.Point([83.277, 17.7009]).buffer(10000).bounds()

# Filter the collection in Earth Engine first
# This is the most reliable way to handle time filtering in XEE
collection = ee.ImageCollection('MODIS/061/MOD44B') \
    .filterDate('2000-01-01', '2023-12-31') \
    .filterBounds(roi)

# Open the collection
ds = xr.open_dataset(
    collection,
    engine='ee',
    geometry=roi,
    scale=250 # MODIS native resolution is ~250m
)

# Crucial: Sort by time for correct charting
ds = ds.sortby('time')

print(ds)
```

## Step 2: Create Spatial Maps

Visualize the state of vegetation in the most recent year compared to the start of the century.

```python
# Select Percent Tree Cover
tree_cover = ds.Percent_Tree_Cover

# Create a side-by-side comparison Map
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

tree_cover.isel(time=0).plot(ax=axes[0], cmap='YlGn', vmin=0, vmax=100)
axes[0].set_title(f'Tree Cover - {tree_cover.time.values[0].year}')

tree_cover.isel(time=-1).plot(ax=axes[1], cmap='YlGn', vmin=0, vmax=100)
axes[1].set_title(f'Tree Cover - {tree_cover.time.values[-1].year}')

plt.tight_layout()
plt.show()
```

## Step 3: Time Series Analysis (Charts)

Plot the average tree cover over time to identify gain or loss trends.

```python
# Calculate spatial mean for each year
tree_cover_ts = tree_cover.mean(dim=['lat', 'lon'])

# Create a professional line chart
plt.figure(figsize=(12, 5))
tree_cover_ts.plot(marker='o', linestyle='-', color='forestgreen', linewidth=2)

# Styling
plt.title('23-Year Trend: Average Percent Tree Cover', fontsize=14)
plt.ylabel('Percent Cover (%)')
plt.xlabel('Year')
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 4: Statistical Graphs (Histograms)

Analyze the distribution of pixels to understand the landscape composition.

```python
# Select the latest year's data
latest_data = ds.isel(time=-1)

plt.figure(figsize=(10, 6))
sns.histplot(latest_data.Percent_Tree_Cover.values.flatten(), color="green", label="Tree Cover", kde=True)
sns.histplot(latest_data.Percent_Non_Tree_Vegetation.values.flatten(), color="orange", label="Non-Tree Veg", kde=True)
sns.histplot(latest_data.Percent_Non_Vegetated.values.flatten(), color="gray", label="Non-Vegetated", kde=True)

plt.title('Landscape Composition Distribution (Latest Year)')
plt.xlabel('Percent Coverage')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

## Step 5: Multi-Band Comparison (Bar Chart)

Compare the mean coverage of different classes.

```python
# Calculate mean of all major classes
means = {
    'Tree Cover': ds.Percent_Tree_Cover.mean().values,
    'Non-Tree Veg': ds.Percent_Non_Tree_Vegetation.mean().values,
    'Non-Vegetated': ds.Percent_Non_Vegetated.mean().values
}

plt.figure(figsize=(8, 5))
plt.bar(means.keys(), means.values(), color=['forestgreen', 'orange', 'gray'])
plt.title('Mean Landscape Composition (2000-2023)')
plt.ylabel('Average Percent Cover')
plt.show()
```

## Key Takeaways

!!! success "Visualization Mastery"
    - **Spatial Maps**: Use `.plot()` with specified time indices to see changes over space.
    - **Charts**: Use `.mean(dim=['lat', 'lon'])` to reduce spatial data into a time series line graph.
    - **Graphs**: Leverage `seaborn` or `matplotlib` directly on `.values.flatten()` for distribution analysis.
    - **VCF Data**: MOD44B is excellent for studying long-term environmental change without the noise of seasonal NDVI.

→ Next: [Deep Learning Architectures](deep-learning-spatial.md)
