# Network and Flow Analysis

Analyze spatial connectivity, optimize paths, and detect communities in geospatial networks using XEE, NetworkX, and OSMnx.

## Overview

This example covers:

1. **Network Fundamentals**: Nodes, Edges, and Adjacency.
2. **Spatial Networks**: Extracting road networks from OSM data.
3. **Network Metrics**: Centrality and Connectivity.
4. **Flow Analysis**: Shortest paths and Flow accumulation concepts.

## Step 1: Initialize Network from OpenStreetMap (OSMnx)

```python
import osmnx as ox
import networkx as nx
import geemap
import matplotlib.pyplot as plt

# Center point (Delhi)
place = "New Delhi, India"

# Download the road network
G = ox.graph_from_place(place, network_type="drive", buffer_dist=2000)

# Project to a local CRS
G_projected = ox.project_graph(G)

# Plot the network
fig, ax = ox.plot_graph(G_projected, node_size=5, edge_linewidth=0.5, edge_color='gray')
```

## Step 2: Network Metrics and Centrality

Identifying the most important nodes in the city infrastructure.

```python
# Calculate Betweenness Centrality
# (How many shortest paths pass through a node)
centrality = nx.betweenness_centrality(G, weight="length")

# Add centrality as a node attribute
nx.set_node_attributes(G, centrality, "centrality")

# Plot with node colors based on centrality
nc = ox.plot.get_node_colors_by_attr(G, "centrality", cmap="plasma")
fig, ax = ox.plot_graph(G, node_color=nc, node_size=15, node_zorder=2, edge_linewidth=0.5)
```

## Step 3: Shortest Path and Flow Analysis

```python
# Define origin and destination nodes (randomly selected)
import random
orig = random.choice(list(G.nodes))
dest = random.choice(list(G.nodes))

# Calculate the shortest path based on length
route = nx.shortest_path(G, orig, dest, weight="length")

# Plot the route
fig, ax = ox.plot_graph_route(G, route, route_linewidth=4, route_color="red")
```

## Step 4: River Flow Analysis (Raster Approach)

In Earth Science, "Flow" often refers to water accumulation in river networks.

```python
import ee
import xarray as xr
import xee

# Load SRTM DEM
roi = ee.Geometry.Rectangle([76.8, 28.5, 77.2, 28.9])
dem = ee.Image("USGS/SRTMGL1_003").clip(roi)

# In EE, we use the Hydrology tools
rivers = ee.HydroEngine.rivers(dem) # Conceptual

# Using XEE for local analysis
ds = xr.open_dataset(dem, engine='ee', geometry=roi, scale=30).compute()

# Local Flow accumulation (Conceptual algorithm)
# Typically uses packages like 'pysheds'
```

## Step 5: Community Detection in Networks

Grouping nodes based on connectivity rather than just proximity.

```python
from networkx.algorithms import community

# Detect communities using greedy modularity
communities = community.greedy_modularity_communities(G)

# Color nodes by community
node_comms = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_comms[node] = i

nx.set_node_attributes(G, node_comms, "community")
nc = ox.plot.get_node_colors_by_attr(G, "community", cmap="tab20")
fig, ax = ox.plot_graph(G, node_color=nc, node_size=10)
```

## Key Takeaways

!!! success "Summary"
    - **Topological Logic**: Networks focus on "how things are connected" rather than "where they are".
    - **Infrastructure**: Network analysis is vital for urban planning and disaster management.
    - **Hydrology**: Combines raster flow models with graph centrality to understand drainage.

→ Next: [Deep Learning Architectures](deep-learning-spatial.md)
