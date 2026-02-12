# Optimization Techniques

Optimizing your cloud-native remote sensing workflows can significantly reduce computation time, memory usage, and cloud costs.

## 1. Spatial and Temporal Filtering

The first rule of optimization is: **Load only what you need.**

- **Spatial**: Use the tightest possible `bbox` or `geometry`. If you're analyzing a city, don't load the whole province.
- **Temporal**: Filter date ranges strictly. If you only need summer months, don't load the full year and then filter in XArray; filter at the STAC/GEE API level.
- **Bands**: Only load the bands required for your index (e.g., Red and NIR for NDVI).

## 2. Chunking Optimization

As discussed in the Dask section, chunking is crucial.

- **Align chunks with data**: Zarr and COG files have internal tiling. Try to make your Dask chunks a multiple of the file's internal tiles.
- **Minimize shuffling**: Avoid operations that require data to move between workers (like global sorting or complex re-gridding) if possible.

## 3. Projection and Alignment

Re-projecting large datasets is computationally expensive.

- **Common CRS**: Try to work in the native CRS of the data. If you must re-project, do it *after* spatial/temporal subsetting.
- **Lazy Reprojection**: Use `odc-stac` or `stackstac` which can handle re-projection lazily during data loading.

## 4. Lazy Operations and Persistence

- **Don't `.compute()` early**: String multiple operations together (Indices → Resampling → Statistics) and run a single `.compute()` at the end.
- **Use `.persist()`**: If you're going to use a filtered dataset multiple times in the same session, use `ds = ds.persist()`. This keeps the data in the memory of the Dask workers, preventing it from being re-loaded or re-calculated from the source.

## 5. Optimized Storage (Zarr)

Zarr is significantly faster than COG for time series analysis because it is optimized for "slicing" along the time dimension.

- **Consolidated Metadata**: Always use `consolidated=True` when writing or reading Zarr. This allows Dask to find all chunks with a single HTTP request rather than hundreds.
- **Compression**: Use Blosc or Zstd compression to reduce data transfer size.

## 6. EECU Optimization (Earth Engine)

When using XEE/Earth Engine:

- **Simplify Geometries**: Complex polygons with thousands of vertices can slow down filtering. Use `geometry.simplify()`.
- **Reduce resolution**: Use a larger `scale` (e.g., 30m instead of 10m) for large area overviews.

## Checklist for High-Performance Code

- [ ] Filtered by BBOX and Date at the API level?
- [ ] Only required bands selected?
- [ ] Dask Dashboard shows workers are busy (not idling)?
- [ ] Memory usage per worker is stable?
- [ ] No small, fragmented chunks?
- [ ] Using Zarr for time-series and COG for spatial composites?
