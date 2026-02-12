# Google Colab Basics

[Google Colab](https://colab.research.google.com/) is a hosted Jupyter notebook environment that allows anyone to run Python code via a web browser. It provides free computation and data storage that can be utilized by your Python code.

## Getting Started

### Creating Your First Notebook

1. Visit [colab.research.google.com](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File → New Notebook**

### Running Code

You can click the **+Code** button to create a new cell and enter a block of code. To run the code, click the **Run Code** button next to the cell, or press `Shift+Enter`.

```python
print('Hello, Cloud Native Remote Sensing!')
```

## Package Management

Colab comes pre-installed with many Python packages. You can use a package by simply importing it:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Installing Additional Packages

If you want to install packages not included by default, use `!pip`:

```python
!pip install rioxarray odc-stac pystac-client
```

!!! tip "Suppress Installation Output"
    Use `%%capture` to hide installation messages:
    ```python
    %%capture
    !pip install rioxarray odc-stac pystac-client
    ```

### Checking Installed Packages

```python
# List all packages
!pip list

# Check specific package version
!pip show xarray
```

## Data Management

Colab provides 100GB of disk space along with your notebook. This can be used to store your data, intermediate outputs, and results.

### Creating Directories

```python
import os

data_folder = 'data'
output_folder = 'output'

if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
```

### Downloading Data

Helper function to download files from URLs:

```python
import requests

def download(url, folder='data'):
    filename = os.path.join(folder, os.path.basename(url))
    if not os.path.exists(filename):
        with requests.get(url, stream=True, allow_redirects=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print('Downloaded', filename)
    return filename

# Example usage
url = 'https://example.com/data.tif'
filepath = download(url)
```

### Reading Data

```python
import geopandas as gpd
import rioxarray as rxr

# Read vector data
gdf = gpd.read_file(filepath)

# Read raster data
raster = rxr.open_rasterio(filepath)
```

## Google Drive Integration

Rather than saving to the temporary Colab machine, you can save to your Google Drive for persistent storage.

### Mounting Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

After running this, click the link and authorize access.

### Working with Drive Files

```python
# Define paths
drive_folder_root = 'MyDrive'
output_folder = 'remote-sensing-outputs'
drive_folder_path = os.path.join('/content/drive', drive_folder_root, output_folder)

# Create folder if it doesn't exist
if not os.path.exists(drive_folder_path):
    os.makedirs(drive_folder_path)

# Save file to Drive
output_file = 'result.tif'
output_path = os.path.join(drive_folder_path, output_file)
raster.rio.to_raster(output_path)
```

### Unmounting Drive

```python
drive.flush_and_unmount()
```

## Runtime Management

### Runtime Types

Colab offers different runtime types:

- **None**: No accelerator (CPU only)
- **GPU**: NVIDIA GPU (T4, P100, or V100)
- **TPU**: Google TPU

To change runtime:

1. Click **Runtime → Change runtime type**
2. Select Hardware accelerator
3. Click **Save**

### Checking Resources

```python
# Check RAM
!cat /proc/meminfo | grep MemTotal

# Check CPU
!cat /proc/cpuinfo | grep "model name" | head -1

# Check GPU (if available)
!nvidia-smi
```

### Session Limits

- **Free tier**: 12-hour maximum runtime
- **Colab Pro**: 24-hour maximum runtime
- Sessions disconnect after 90 minutes of inactivity

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | `Ctrl+Enter` or `Shift+Enter` |
| Insert cell above | `Ctrl+M A` |
| Insert cell below | `Ctrl+M B` |
| Delete cell | `Ctrl+M D` |
| Convert to code | `Ctrl+M Y` |
| Convert to text | `Ctrl+M M` |
| Show shortcuts | `Ctrl+M H` |

## Magic Commands

Colab supports IPython magic commands:

```python
# Time execution
%%time
result = expensive_computation()

# Time multiple runs
%%timeit
quick_computation()

# Run shell commands
!ls -la

# Change directory
%cd /content/data

# Show current directory
%pwd

# Load external Python file
%load script.py

# Run external Python file
%run script.py
```

## Visualization

### Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
raster.plot(ax=ax, cmap='viridis')
plt.title('Satellite Image')
plt.show()
```

### Interactive Plots

```python
# Install plotly
!pip install plotly

import plotly.express as px

# Create interactive plot
fig = px.scatter(df, x='lon', y='lat', color='value')
fig.show()
```

## Forms and Widgets

Create interactive forms:

```python
#@title Configuration { run: "auto" }
year = 2023 #@param {type:"slider", min:2015, max:2024, step:1}
cloud_cover = 30 #@param {type:"slider", min:0, max:100, step:5}
region = "Global" #@param ["Global", "North America", "Europe", "Asia"]

print(f"Year: {year}")
print(f"Max Cloud Cover: {cloud_cover}%")
print(f"Region: {region}")
```

## Sharing Notebooks

### Save to GitHub

1. Click **File → Save a copy in GitHub**
2. Authorize GitHub access
3. Select repository and branch
4. Add commit message
5. Click **OK**

### Share Link

1. Click **Share** button (top right)
2. Set permissions:
   - **Viewer**: Can view only
   - **Commenter**: Can comment
   - **Editor**: Can edit
3. Copy link

### Download Notebook

```python
# Download as .ipynb
# File → Download → Download .ipynb

# Download as .py
# File → Download → Download .py
```

## Best Practices

### 1. Save Frequently

Colab auto-saves, but manually save important work:

- `Ctrl+S` or **File → Save**

### 2. Use Version Control

```python
# Save checkpoint
# File → Save a copy in Drive
# File → Revision history
```

### 3. Organize Code

```python
# Use markdown cells for documentation
# Use code cells for executable code
# Group related operations together
```

### 4. Clear Outputs

Before sharing:

- **Edit → Clear all outputs**

### 5. Restart Runtime

If experiencing issues:

- **Runtime → Restart runtime**

## Working with Large Datasets

### Streaming Data

Don't download entire datasets:

```python
import xarray as xr

# Stream from cloud storage
ds = xr.open_dataset(
    'https://example.com/large_dataset.nc',
    chunks={'time': 10}
)
```

### Using Dask

```python
from dask.distributed import Client

# Start local Dask cluster
client = Client()

# View dashboard
client
```

### Viewing Dask Dashboard in Colab

```python
from google.colab import output

port_to_expose = 8787  # Dask dashboard port
print(output.eval_js(f'google.colab.kernel.proxyPort({port_to_expose})'))
```

## Troubleshooting

### Common Issues

**1. Runtime Disconnected**

```python
# Solution: Reconnect
# Runtime → Reconnect
```

**2. Out of Memory**

```python
# Solution: Use smaller chunks or restart runtime
# Runtime → Restart runtime
```

**3. Package Import Errors**

```python
# Solution: Reinstall package
!pip install --upgrade --force-reinstall package-name
```

**4. Drive Mount Issues**

```python
# Solution: Unmount and remount
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

## Advanced Features

### GPU Acceleration

```python
import tensorflow as tf

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Use GPU for computation
with tf.device('/GPU:0'):
    # Your GPU-accelerated code here
    pass
```

### Parallel Processing

```python
from joblib import Parallel, delayed

def process_file(filename):
    # Processing logic
    return result

# Process files in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_file)(f) for f in files
)
```

### Custom Snippets

Create reusable code snippets:

1. **Tools → Command palette**
2. Search "Insert code snippet"
3. Create custom snippet

## Example Workflow

Complete example of a typical remote sensing workflow in Colab:

```python
# 1. Install packages
%%capture
!pip install rioxarray pystac-client odc-stac

# 2. Import libraries
import xarray as xr
import rioxarray as rxr
from pystac_client import Client
import matplotlib.pyplot as plt

# 3. Search for data
catalog = Client.open('https://earth-search.aws.element84.com/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[lon_min, lat_min, lon_max, lat_max],
    datetime='2023-01-01/2023-12-31'
)
items = search.item_collection()

# 4. Load data
from odc.stac import load as stac_load
ds = stac_load(
    items,
    bands=['red', 'green', 'blue', 'nir'],
    resolution=10
)

# 5. Process
ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)

# 6. Visualize
fig, ax = plt.subplots(figsize=(10, 8))
ndvi.isel(time=0).plot(ax=ax, cmap='RdYlGn')
plt.title('NDVI')
plt.show()

# 7. Save to Drive
from google.colab import drive
drive.mount('/content/drive')
output_path = '/content/drive/MyDrive/ndvi_result.nc'
ndvi.to_netcdf(output_path)
```

## Next Steps

Now that you're familiar with Google Colab:

→ Continue to [XArray Basics](../fundamentals/xarray-basics.md)

## Additional Resources

- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Notebooks Gallery](https://colab.research.google.com/notebooks/)
- [Markdown Guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
- [Data Science Snippets](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)

## Quick Reference Card

```python
# Essential Colab Commands

# Package management
!pip install package-name
!pip list
!pip show package-name

# File system
!ls
!pwd
!mkdir folder_name
!rm file_name

# System information
!cat /proc/meminfo
!nvidia-smi

# Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Magic commands
%%time
%%timeit
%pwd
%cd directory
```
