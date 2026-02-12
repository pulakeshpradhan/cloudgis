# Multi-temporal Analysis with XEE

Advanced time series analysis techniques for detecting patterns, trends, and anomalies in Earth Engine data.

## Overview

This example demonstrates:

- Seasonal decomposition
- Trend analysis
- Anomaly detection
- Phenology extraction
- Harmonic regression

**Dataset**: MODIS NDVI for phenology analysis

## Step 1: Load Multi-year Time Series

```python
import ee
import xarray as xr
import xee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression

# Initialize Earth Engine
ee.Initialize(project='spatialgeography')

# Define region (agricultural area)
roi = ee.Geometry.Point([82.5, 27.0]).buffer(10000)

# Load MODIS NDVI (2020-2023)
collection = ee.ImageCollection('MODIS/061/MOD13A2') \
    .filterBounds(roi) \
    .filterDate('2020-01-01', '2023-12-31') \
    .select('NDVI')

print(f"Collection size: {collection.size().getInfo()} images")

# Load with XEE
ds = xr.open_dataset(
    collection,
    engine='ee',
    geometry=roi,
    scale=500,  # MODIS resolution
    crs='EPSG:4326'
)

# Sort by time (critical for time series!)
ds = ds.sortby('time')

# Scale NDVI (MODIS NDVI has scale factor of 0.0001)
ds['NDVI'] = ds['NDVI'] * 0.0001

# Compute
ds = ds.compute()

print(f"Time series length: {len(ds.time)}")
print(f"Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
```

## Step 2: Calculate Spatial Mean Time Series

```python
# Calculate mean NDVI over the region
ndvi_ts = ds.NDVI.mean(dim=['lon', 'lat'])

# Convert to pandas for easier manipulation
ndvi_df = ndvi_ts.to_dataframe(name='NDVI').reset_index()
ndvi_df['date'] = pd.to_datetime(ndvi_df['time'])

# Plot raw time series
plt.figure(figsize=(14, 6))
plt.plot(ndvi_df['date'], ndvi_df['NDVI'], 'o-', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('MODIS NDVI Time Series (2020-2023)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Step 3: Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Resample to regular monthly intervals
ndvi_monthly = ndvi_df.set_index('date')['NDVI'].resample('M').mean()

# Perform seasonal decomposition
decomposition = seasonal_decompose(ndvi_monthly, model='additive', period=12)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

for ax in axes:
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 4: Trend Analysis

```python
# Linear trend
time_numeric = np.arange(len(ndvi_monthly))
X = time_numeric.reshape(-1, 1)
y = ndvi_monthly.values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)

# Calculate trend statistics
slope = model.coef_[0]
r_squared = model.score(X, y)

# Statistical significance
_, p_value = stats.pearsonr(time_numeric, y)

print(f"Trend slope: {slope:.6f} NDVI/month")
print(f"R-squared: {r_squared:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Trend is {'significant' if p_value < 0.05 else 'not significant'} at α=0.05")

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(ndvi_monthly.index, ndvi_monthly.values, 'o-', label='Observed', alpha=0.7)
plt.plot(ndvi_monthly.index, trend_line, 'r--', linewidth=2, label=f'Trend (slope={slope:.6f})')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('NDVI Trend Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Step 5: Anomaly Detection

```python
# Calculate climatology (multi-year average for each month)
ndvi_df['month'] = ndvi_df['date'].dt.month
climatology = ndvi_df.groupby('month')['NDVI'].mean()

# Calculate anomalies
ndvi_df['climatology'] = ndvi_df['month'].map(climatology)
ndvi_df['anomaly'] = ndvi_df['NDVI'] - ndvi_df['climatology']

# Calculate standard deviation for threshold
std_anomaly = ndvi_df['anomaly'].std()
threshold = 2 * std_anomaly

# Identify significant anomalies
ndvi_df['significant_anomaly'] = np.abs(ndvi_df['anomaly']) > threshold

# Plot anomalies
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Time series with climatology
axes[0].plot(ndvi_df['date'], ndvi_df['NDVI'], 'o-', label='Observed', alpha=0.7)
axes[0].plot(ndvi_df['date'], ndvi_df['climatology'], 'g--', linewidth=2, label='Climatology')
axes[0].fill_between(ndvi_df['date'], 
                      ndvi_df['climatology'] - threshold,
                      ndvi_df['climatology'] + threshold,
                      alpha=0.2, color='gray', label='±2σ threshold')
axes[0].set_ylabel('NDVI')
axes[0].set_title('NDVI with Climatology')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Anomalies
colors = ['red' if x else 'blue' for x in ndvi_df['significant_anomaly']]
axes[1].bar(ndvi_df['date'], ndvi_df['anomaly'], color=colors, alpha=0.7)
axes[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
axes[1].axhline(y=-threshold, color='r', linestyle='--')
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Anomaly')
axes[1].set_title('NDVI Anomalies (Red = Significant)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print significant anomalies
print("\nSignificant Anomalies:")
significant = ndvi_df[ndvi_df['significant_anomaly']]
for _, row in significant.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d')}: {row['anomaly']:.3f}")
```

## Step 6: Phenology Extraction

```python
# Extract phenological metrics for each year
def extract_phenology(year_data):
    """Extract key phenological dates and values."""
    
    # Find peak (maximum NDVI)
    peak_idx = year_data['NDVI'].idxmax()
    peak_date = year_data.loc[peak_idx, 'date']
    peak_value = year_data.loc[peak_idx, 'NDVI']
    
    # Find start of season (first date NDVI > threshold)
    threshold = year_data['NDVI'].quantile(0.3)
    sos_candidates = year_data[year_data['NDVI'] > threshold]
    sos_date = sos_candidates.iloc[0]['date'] if len(sos_candidates) > 0 else None
    
    # Find end of season (last date NDVI > threshold)
    eos_date = sos_candidates.iloc[-1]['date'] if len(sos_candidates) > 0 else None
    
    # Calculate length of growing season
    los = (eos_date - sos_date).days if sos_date and eos_date else None
    
    return {
        'year': year_data['date'].dt.year.iloc[0],
        'sos': sos_date,
        'peak': peak_date,
        'eos': eos_date,
        'peak_ndvi': peak_value,
        'los_days': los
    }

# Extract phenology for each year
phenology_results = []
for year in range(2020, 2024):
    year_data = ndvi_df[ndvi_df['date'].dt.year == year]
    if len(year_data) > 0:
        pheno = extract_phenology(year_data)
        phenology_results.append(pheno)

pheno_df = pd.DataFrame(phenology_results)
print("\nPhenology Metrics:")
print(pheno_df)

# Visualize phenology
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, year in enumerate(range(2020, 2024)):
    ax = axes[idx // 2, idx % 2]
    year_data = ndvi_df[ndvi_df['date'].dt.year == year]
    pheno = pheno_df[pheno_df['year'] == year].iloc[0]
    
    # Plot NDVI
    ax.plot(year_data['date'], year_data['NDVI'], 'o-', alpha=0.7)
    
    # Mark phenological events
    if pd.notna(pheno['sos']):
        ax.axvline(pheno['sos'], color='g', linestyle='--', label='Start of Season')
    if pd.notna(pheno['peak']):
        ax.axvline(pheno['peak'], color='r', linestyle='--', label='Peak')
    if pd.notna(pheno['eos']):
        ax.axvline(pheno['eos'], color='orange', linestyle='--', label='End of Season')
    
    ax.set_title(f'{year} - Growing Season: {pheno["los_days"]} days')
    ax.set_ylabel('NDVI')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 7: Harmonic Analysis

```python
# Fit harmonic model (annual + semi-annual cycles)
def harmonic_model(t, a0, a1, b1, a2, b2):
    """Harmonic model with annual and semi-annual components."""
    omega = 2 * np.pi / 365.25  # Annual frequency
    return (a0 + 
            a1 * np.cos(omega * t) + b1 * np.sin(omega * t) +
            a2 * np.cos(2 * omega * t) + b2 * np.sin(2 * omega * t))

from scipy.optimize import curve_fit

# Prepare data
days_since_start = (ndvi_df['date'] - ndvi_df['date'].min()).dt.days.values
ndvi_values = ndvi_df['NDVI'].values

# Fit harmonic model
popt, _ = curve_fit(harmonic_model, days_since_start, ndvi_values)

# Generate fitted values
fitted_values = harmonic_model(days_since_start, *popt)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(ndvi_df['date'], ndvi_values, 'o', alpha=0.5, label='Observed')
plt.plot(ndvi_df['date'], fitted_values, 'r-', linewidth=2, label='Harmonic Fit')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('Harmonic Analysis (Annual + Semi-annual)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate R-squared
ss_res = np.sum((ndvi_values - fitted_values) ** 2)
ss_tot = np.sum((ndvi_values - np.mean(ndvi_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Harmonic model R-squared: {r_squared:.3f}")
```

## Step 8: Spatial Patterns of Trends

```python
# Calculate pixel-wise trends
def calculate_pixel_trend(pixel_ts):
    """Calculate linear trend for a single pixel."""
    time_numeric = np.arange(len(pixel_ts))
    mask = ~np.isnan(pixel_ts)
    
    if mask.sum() < 3:
        return np.nan
    
    slope, _, _, p_value, _ = stats.linregress(time_numeric[mask], pixel_ts[mask])
    return slope if p_value < 0.05 else 0  # Only significant trends

# Apply to all pixels
trend_map = xr.apply_ufunc(
    calculate_pixel_trend,
    ds.NDVI,
    input_core_dims=[['time']],
    vectorize=True
)

# Visualize spatial trends
plt.figure(figsize=(12, 10))
trend_map.plot(cmap='RdBu_r', center=0, vmin=-0.01, vmax=0.01)
plt.title('Spatial Pattern of NDVI Trends (2020-2023)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()
```

## Step 9: Export Results

```python
# Create comprehensive results dataset
results = xr.Dataset({
    'ndvi_mean': ds.NDVI.mean(dim='time'),
    'ndvi_std': ds.NDVI.std(dim='time'),
    'ndvi_trend': trend_map,
    'ndvi_max': ds.NDVI.max(dim='time'),
    'ndvi_min': ds.NDVI.min(dim='time')
})

# Save results
results.to_netcdf('multitemporal_analysis_results.nc')
print("Spatial results saved to multitemporal_analysis_results.nc")

# Save time series analysis
analysis_df = pd.DataFrame({
    'date': ndvi_df['date'],
    'ndvi': ndvi_df['NDVI'],
    'climatology': ndvi_df['climatology'],
    'anomaly': ndvi_df['anomaly'],
    'significant_anomaly': ndvi_df['significant_anomaly']
})
analysis_df.to_csv('timeseries_analysis.csv', index=False)
print("Time series analysis saved to timeseries_analysis.csv")

# Save phenology metrics
pheno_df.to_csv('phenology_metrics.csv', index=False)
print("Phenology metrics saved to phenology_metrics.csv")
```

## Step 10: Interactive Visualization

```python
# Create interactive plot with plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('NDVI Time Series', 'Anomalies', 'Trend Components'),
    vertical_spacing=0.1
)

# Time series
fig.add_trace(
    go.Scatter(x=ndvi_df['date'], y=ndvi_df['NDVI'], 
               mode='lines+markers', name='NDVI'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=ndvi_df['date'], y=ndvi_df['climatology'],
               mode='lines', name='Climatology', line=dict(dash='dash')),
    row=1, col=1
)

# Anomalies
colors = ['red' if x else 'blue' for x in ndvi_df['significant_anomaly']]
fig.add_trace(
    go.Bar(x=ndvi_df['date'], y=ndvi_df['anomaly'],
           marker_color=colors, name='Anomaly'),
    row=2, col=1
)

# Trend
fig.add_trace(
    go.Scatter(x=ndvi_monthly.index, y=ndvi_monthly.values,
               mode='markers', name='Monthly Mean'),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=ndvi_monthly.index, y=trend_line,
               mode='lines', name='Trend', line=dict(color='red', width=2)),
    row=3, col=1
)

fig.update_layout(height=900, showlegend=True, title_text="Multi-temporal NDVI Analysis")
fig.show()
```

## Key Takeaways

!!! success "What You Learned"
    - Loading and processing multi-year time series with XEE
    - Seasonal decomposition of time series
    - Trend analysis with statistical significance
    - Anomaly detection using climatology
    - Phenology extraction (SOS, EOS, peak)
    - Harmonic regression for seasonal patterns
    - Spatial trend mapping
    - Comprehensive result export and visualization

## Next Steps

→ Continue to [Reference Section](../reference/xarray-api.md)

## Additional Resources

- [Time Series Analysis in Python](https://www.statsmodels.org/stable/tsa.html)
- [Phenology Extraction](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/phenology)
- [Harmonic Analysis](https://developers.google.com/earth-engine/tutorials/community/time-series-modeling)
