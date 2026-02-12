# Cloud Computing for Remote Sensing

Modern remote sensing has shifted from "download and process" to "process in the cloud." This page covers the major cloud platforms used in remote sensing and their advantages.

## Major Cloud Providers

### 1. Google Cloud Platform (GCP)

- **Data**: Google Earth Engine (GEE) Catalog, Cloud Storage.
- **Compute**: Compute Engine (VMs), Google Kubernetes Engine (GKE), Vertex AI for Machine Learning.
- **Strength**: Seamless integration with Earth Engine and massive public datasets.

### 2. Amazon Web Services (AWS)

- **Data**: Amazon S3 (hosts the Sentinel-2 and Landsat repositories), Registry of Open Data.
- **Compute**: EC2, SageMaker (ML), Lambda (Serverless).
- **Strength**: Most cloud-native data (COGs) is natively hosted on AWS S3, enabling ultra-fast range requests.

### 3. Microsoft Azure

- **Data**: Azure Blob Storage, Planetary Computer.
- **Compute**: Virtual Machines, Azure Batch, Azure ML.
- **Strength**: The **Planetary Computer** is a best-in-class STAC-based catalog for open geospatial data.

## Serverless vs. Managed vs. Infrastructure

### Infrastructure as a Service (IaaS)

Renting Virtual Machines (e.g., AWS EC2, GCP Compute Engine).

- **Pro**: Full control over environment.
- **Con**: You manage security updates and scaling.

### Managed Platforms (PaaS)

Platforms that manage the compute cluster for you (e.g., Google Earth Engine, Microsoft Planetary Computer).

- **Pro**: Focus on science, not infrastructure.
- **Con**: Less control over underlying hardware/versions.

### Serverless (FaaS)

Running functions without managing servers (e.g., AWS Lambda, Google Cloud Functions).

- **Pro**: Pay-per-execution, scales to zero.
- **Con**: Execution time limits (usually 15 minutes). Perfect for small STAC queries or metadata processing.

## Cloud-Native Workflows

1. **Discovery**: Use **STAC** to find data on the cloud.
2. **Access**: Use **HTTP Range Requests** to stream only the pixels you need from COGs or Zarr.
3. **Process**: Use **Dask** or server-side reducers (in Earth Engine) to process data in the cloud.
4. **Storage**: Save results back to cloud storage (S3/GCS/Blob) in Zarr or COG format.

## Cost Considerations

When working in the cloud, be mindful of:

- **Compute costs**: Active VMs or Dask workers charge per second/minute.
- **Storage costs**: Monthly charges for keeping data in S3/Cloud Storage.
- **Data Transfer (Egress)**: Downloading large amounts of data *out* of the cloud region to your local machine can be expensive. Always try to process data in the same region where it is stored.
