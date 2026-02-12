# Setup Environment

This guide will help you set up your environment for cloud-native remote sensing with Python. We'll cover both Google Colab (recommended for beginners) and local installation.

## Option 1: Google Colab (Recommended)

Google Colab is a free Jupyter notebook environment that runs in the cloud. It's perfect for this course because:

- ✅ No installation required
- ✅ Free GPU/TPU access
- ✅ Pre-installed common libraries
- ✅ Easy sharing and collaboration
- ✅ Persistent storage with Google Drive

### Getting Started with Colab

1. **Access Google Colab**
   - Visit [colab.research.google.com](https://colab.research.google.com/)
   - Sign in with your Google account

2. **Create a New Notebook**
   - Click "New Notebook" or File → New Notebook
   - Rename your notebook (File → Rename)

3. **Install Required Packages**

```python
%%capture
!pip install xarray rioxarray pystac-client odc-stac dask zarr xee earthengine-api
```

### Colab Pro (Optional)

For intensive work, consider Colab Pro ($9.99/month):

- Longer runtimes
- More memory
- Faster GPUs
- Background execution

## Option 2: Local Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- 8GB+ RAM recommended
- Git (for version control)

### Using Conda (Recommended)

```bash
# Create a new environment
conda create -n remote-sensing python=3.10

# Activate the environment
conda activate remote-sensing

# Install packages
conda install -c conda-forge xarray dask netCDF4 bottleneck numpy pandas matplotlib

# Install additional packages with pip
pip install rioxarray pystac-client odc-stac zarr xee earthengine-api
```

### Using pip

```bash
# Create virtual environment
python -m venv remote-sensing-env

# Activate (Windows)
remote-sensing-env\Scripts\activate

# Activate (Linux/Mac)
source remote-sensing-env/bin/activate

# Install packages
pip install xarray[complete] rioxarray pystac-client odc-stac dask[complete] zarr xee earthengine-api matplotlib jupyter
```

## Required Packages

### Core Libraries

| Package | Purpose | Installation |
|---------|---------|--------------|
| **xarray** | Multi-dimensional arrays | `pip install xarray` |
| **rioxarray** | Geospatial extensions | `pip install rioxarray` |
| **dask** | Parallel computing | `pip install dask[complete]` |
| **zarr** | Cloud-optimized storage | `pip install zarr` |

### Data Access

| Package | Purpose | Installation |
|---------|---------|--------------|
| **pystac-client** | STAC API client | `pip install pystac-client` |
| **odc-stac** | Load STAC to XArray | `pip install odc-stac` |
| **xee** | Earth Engine integration | `pip install xee` |
| **earthengine-api** | Earth Engine Python API | `pip install earthengine-api` |

### Visualization

| Package | Purpose | Installation |
|---------|---------|--------------|
| **matplotlib** | Static plots | `pip install matplotlib` |
| **hvplot** | Interactive plots | `pip install hvplot` |
| **folium** | Interactive maps | `pip install folium` |

### Optional but Useful

```bash
pip install geopandas shapely fiona pyproj jupyter jupyterlab ipywidgets
```

## Verification

Test your installation with this script:

```python
import sys
print(f"Python version: {sys.version}")

# Test imports
try:
    import xarray as xr
    print(f"✓ xarray {xr.__version__}")
except ImportError:
    print("✗ xarray not found")

try:
    import rioxarray
    print(f"✓ rioxarray installed")
except ImportError:
    print("✗ rioxarray not found")

try:
    import pystac_client
    print(f"✓ pystac-client installed")
except ImportError:
    print("✗ pystac-client not found")

try:
    import dask
    print(f"✓ dask {dask.__version__}")
except ImportError:
    print("✗ dask not found")

try:
    import zarr
    print(f"✓ zarr {zarr.__version__}")
except ImportError:
    print("✗ zarr not found")

try:
    import xee
    print(f"✓ xee installed")
except ImportError:
    print("✗ xee not found")

print("\n✅ All core packages installed successfully!")
```

## Setting Up Earth Engine (Optional)

If you want to use Google Earth Engine:

1. **Create an Earth Engine Account**
   - Visit [earthengine.google.com](https://earthengine.google.com/)
   - Sign up for access

2. **Authenticate**

```python
import ee

# Authenticate (first time only)
ee.Authenticate()

# Initialize
ee.Initialize(project='spatialgeography')
```

## IDE Recommendations

### Jupyter Lab

Best for interactive data exploration:

```bash
pip install jupyterlab
jupyter lab
```

### VS Code

Great for development with extensions:

- Python
- Jupyter
- Remote Development
- GitLens

### PyCharm

Full-featured IDE for Python development

## Cloud Platforms

### AWS

- EC2 for compute
- S3 for storage
- SageMaker for ML

### Google Cloud

- Compute Engine
- Cloud Storage
- Earth Engine

### Microsoft Azure

- Virtual Machines
- Blob Storage
- Planetary Computer

## Directory Structure

Organize your projects:

```
remote-sensing-project/
├── data/              # Local data cache
├── notebooks/         # Jupyter notebooks
├── scripts/           # Python scripts
├── outputs/           # Results and figures
├── environment.yml    # Conda environment
└── requirements.txt   # Pip requirements
```

## Best Practices

### 1. Use Virtual Environments

Always isolate your project dependencies:

```bash
conda create -n project-name python=3.10
```

### 2. Pin Package Versions

Create reproducible environments:

```txt
# requirements.txt
xarray==2023.12.0
dask==2023.12.0
zarr==2.16.1
```

### 3. Use Git for Version Control

```bash
git init
git add .
git commit -m "Initial commit"
```

### 4. Document Your Environment

```bash
# Export conda environment
conda env export > environment.yml

# Export pip requirements
pip freeze > requirements.txt
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```python
# Solution: Reinstall package
pip install --upgrade --force-reinstall package-name
```

**2. Memory Errors**

```python
# Solution: Use Dask chunking
ds = xr.open_dataset('file.nc', chunks={'time': 10})
```

**3. GDAL/Rasterio Issues**

```bash
# Solution: Use conda for GDAL
conda install -c conda-forge gdal rasterio
```

## Performance Tips

### 1. Configure Dask

```python
import dask
dask.config.set({'array.slicing.split_large_chunks': True})
```

### 2. Set Chunk Sizes

```python
# Good chunking
ds = xr.open_zarr('data.zarr', chunks={'time': 10, 'x': 512, 'y': 512})

# Avoid too small or too large chunks
```

### 3. Use Local Dask Cluster

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4, threads_per_worker=2)
client = Client(cluster)
```

## Next Steps

Now that your environment is set up:

→ Continue to [Google Colab Basics](colab-basics.md)

## Additional Resources

- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)

## Quick Reference

### Activate Environment

```bash
# Conda
conda activate remote-sensing

# Pip/venv
source remote-sensing-env/bin/activate  # Linux/Mac
remote-sensing-env\Scripts\activate     # Windows
```

### Update Packages

```bash
# Conda
conda update --all

# Pip
pip install --upgrade package-name
```

### List Installed Packages

```bash
# Conda
conda list

# Pip
pip list
```
