# Open GIS - Cloud Native Geospatial Analysis

This repository contains a comprehensive **Jupyter Book** guide to cloud-native remote sensing using Python, XArray, STAC, Dask, Zarr, and XEE.

## 🚀 About This Project

This is a **Jupyter Book** project that teaches open-source, patent-friendly, cloud-based remote sensing workflows. All content is in **Jupyter Notebook (`.ipynb`)** format, making it interactive and executable.

### Key Features

- ✅ **100% Jupyter Notebooks** - All documentation is in `.ipynb` format
- ✅ **Interactive Code Examples** - Run code directly in notebooks
- ✅ **Cloud-Native Workflows** - XArray, STAC, Dask, Zarr, XEE
- ✅ **GitHub Pages Deployment** - Automatic publishing via GitHub Actions
- ✅ **Open Source** - No vendor lock-in, fully reproducible

## 📚 View the Book

**Live Site:** [https://pulakeshpradhan.github.io/cloudgis/](https://pulakeshpradhan.github.io/cloudgis/)

## 🛠️ Building Locally

### Prerequisites

```bash
pip install jupyter-book ghp-import jupytext
```

### Build the Book

```bash
jb build .
```

The HTML output will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build

```bash
jb clean .
jb build .
```

## 📂 Project Structure

```
cloud-native-remote-sensing/
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents
├── docs/                    # All content (100% .ipynb files)
│   ├── index.ipynb         # Homepage
│   ├── getting-started/    # Setup and introduction
│   ├── fundamentals/       # Core libraries
│   ├── processing/         # Data processing workflows
│   ├── advanced/           # Advanced topics
│   ├── examples/           # Practical examples
│   ├── reference/          # API reference
│   └── resources/          # Additional resources
├── _build/                  # Generated HTML (gitignored)
└── .github/workflows/       # GitHub Actions for deployment
```

## 🌐 Publishing to GitHub Pages

The book is automatically published to GitHub Pages on every push to `main` via GitHub Actions.

### Manual Publishing

```bash
ghp-import -n -p -f _build/html
```

## 📖 Content Overview

- **Getting Started**: Environment setup, Google Colab basics
- **Fundamentals**: XArray, STAC, Dask, Zarr, XEE
- **Data Processing**: Spectral indices, cloud masking, time series
- **Advanced Topics**: Scaling with Dask, cloud computing, optimization
- **Practical Examples**: NDVI analysis, classification, change detection
- **Reference**: API documentation and best practices
- **Resources**: Datasets, tools, further reading

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes (remember: all content must be `.ipynb` files)
4. Submit a pull request

## 📝 License

This content is available under the Creative Commons Attribution 4.0 International License.

## 🔗 Links

- **Repository**: [https://github.com/pulakeshpradhan/cloudgis](https://github.com/pulakeshpradhan/cloudgis)
- **Jupyter Book Documentation**: [https://jupyterbook.org](https://jupyterbook.org)
- **Issues**: [https://github.com/pulakeshpradhan/cloudgis/issues](https://github.com/pulakeshpradhan/cloudgis/issues)

---

**Note**: This project uses **Jupyter Book** for building and **GitHub Pages** for hosting. All documentation files are in **Jupyter Notebook (`.ipynb`)** format - there are NO `.md` files in the docs directory.
