# Cloud Native Remote Sensing with Python

A comprehensive course on cloud-native remote sensing using Python, covering XArray, STAC, Dask, Zarr, and XEE for efficient processing of Earth observation data.

## 🎯 Overview

This course provides a structured introduction to modern cloud-native technologies for remote sensing:

- **XArray**: Multi-dimensional labeled arrays for satellite imagery
- **STAC**: Standardized data discovery and access
- **Dask**: Parallel and distributed computing
- **Zarr**: Cloud-optimized array storage
- **XEE**: XArray Earth Engine integration

## 📚 Course Content

### Getting Started

- Introduction to cloud-native remote sensing
- Environment setup (Google Colab & local)
- Google Colab basics

### Fundamentals

- XArray basics and operations
- STAC catalogs and data discovery
- Dask for parallel computing
- Zarr for cloud-optimized storage
- XEE for Earth Engine integration

### Data Processing

- Calculating spectral indices (NDVI, NDWI, SAVI, EVI)
- Cloud masking techniques
- Time series extraction and analysis
- Data aggregation methods

### Advanced Topics

- Scaling with Dask clusters
- Cloud computing platforms
- Planetary Computer integration
- Performance optimization

### Practical Examples

- NDVI analysis workflows
- Land cover classification
- Change detection
- Multi-temporal analysis

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

No installation required! Just click the "Open in Colab" buttons in the course materials.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cloud-native-remote-sensing.git
cd cloud-native-remote-sensing

# Create conda environment
conda create -n remote-sensing python=3.10
conda activate remote-sensing

# Install dependencies
pip install -r requirements.txt

# Serve the documentation locally
mkdocs serve
```

Visit `http://localhost:8000` to view the course.

## 📖 Building the Documentation

```bash
# Install MkDocs and dependencies
pip install mkdocs mkdocs-material

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## 🛠️ Technologies Used

- **Python 3.9+**
- **XArray** - N-dimensional labeled arrays
- **Dask** - Parallel computing
- **Zarr** - Cloud-optimized storage
- **STAC** - Spatiotemporal asset catalogs
- **XEE** - Earth Engine integration
- **MkDocs** - Documentation framework
- **Material for MkDocs** - Documentation theme

## 📦 Key Dependencies

```
xarray>=2023.1.0
dask[complete]>=2023.1.0
zarr>=2.13.0
pystac-client>=0.5.0
odc-stac>=0.3.0
rioxarray>=0.13.0
xee>=0.0.12
earthengine-api>=0.1.300
```

## 🎓 Learning Path

1. **Beginners**: Start with Getting Started → Fundamentals
2. **Intermediate**: Focus on Data Processing → Practical Examples
3. **Advanced**: Dive into Advanced Topics and optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original course content adapted from [Spatial Thoughts](https://spatialthoughts.com/)
- XArray community and documentation
- Pangeo project for cloud-native geoscience tools
- Google Earth Engine team
- Element84 for Earth Search STAC API

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🌟 Star History

If you find this course helpful, please consider giving it a star! ⭐

## 📚 Additional Resources

- [XArray Documentation](https://docs.xarray.dev/)
- [STAC Specification](https://stacspec.org/)
- [Dask Documentation](https://docs.dask.org/)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [Pangeo Community](https://pangeo.io/)
- [Earth Engine](https://earthengine.google.com/)

---

Made with ❤️ for the remote sensing community
