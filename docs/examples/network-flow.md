# Python-First Network and Flow Analysis

Perform advanced spatial network analysis, path optimization, and community detection using `OSMnx` and `NetworkX`. This workflow prioritizes local, reproducible computation over proprietary cloud-based black-box solutions.

## Overview

Modern urban and environmental analysis requires understanding not just *where* things are, but how they are *connected*. By using Python's native network libraries, you ensure:

1. **Transparency**: Every step of the routing and centrality algorithm is visible.
2. **Intellectual Property**: Custom network discoveries are your own, suitable for scientific publication and patenting.
3. **Performance**: Large-scale graph operations are handled efficiently by your local CPU/GPU resources.

## Step 1: Initialize Network from OpenStreetMap (OSMnx)

We bridge the gap between "Open Data" (OSM) and "Open Analysis" (Python) by downloading data directly into a local graph model.

```python
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Define area (e.g., New Delhi)
place = "New Delhi, India"

# Download the road network locally
G = ox.graph_from_place(place, network_type="drive", buffer_dist=2000)

# Project to a local UTM CRS for accurate distance calculations
G_projected = ox.project_graph(G)

# Plot the network locally
fig, ax = ox.plot_graph(G_projected, node_size=5, edge_linewidth=0.5, edge_color='#666666', bgcolor='w')
```

## Step 2: Topology and Centrality

Identify critical hubs in city infrastructure by calculating how many shortest paths pass through each intersection.

```python
# Calculate Betweenness Centrality
# This measures the 'gateway' importance of each node
centrality = nx.betweenness_centrality(G_projected, weight="length", normalized=True)

# Map centrality back to nodes
nx.set_node_attributes(G_projected, centrality, "centrality")

# Visualize the 'hottest' hubs in the network
nc = ox.plot.get_node_colors_by_attr(G_projected, "centrality", cmap="inferno")
fig, ax = ox.plot_graph(G_projected, node_color=nc, node_size=20, edge_linewidth=0.3)
```

## Step 3: Optimal Path Finding

Unlike web-map routing APIs, you can customize the "cost" of every edge (time, speed, traffic, or even greenness).

```python
# Select random nodes for origin and destination
import random
nodes = list(G_projected.nodes)
orig, dest = random.sample(nodes, 2)

# Calculate the shortest path using Dijkstra's algorithm (local execution)
route = nx.shortest_path(G_projected, orig, dest, weight="length")

# Plot the route on your local network
fig, ax = ox.plot_graph_route(G_projected, route, route_linewidth=5, route_color="red")
```

## Step 4: Community Detection

Group neighborhoods based on their connectivity patterns rather than just geographical distance.

```python
from networkx.algorithms import community

# Detect communities using greedy modularity maximization
communities = community.greedy_modularity_communities(G_projected)

# Apply community colors
node_comms = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_comms[node] = i

nx.set_node_attributes(G_projected, node_comms, "community")
nc = ox.plot.get_node_colors_by_attr(G_projected, "community", cmap="tab20")
fig, ax = ox.plot_graph(G_projected, node_color=nc, node_size=15)
```

## Key Takeaways

!!! success "Why Local Networks?"
    - **Custom Costs**: You aren't limited to "driving/walking" time; you can route based on any attribute (emissions, safety, uphill/downhill).
    - **Algorithmic Freedom**: Implement your own discovery methods for network science.
    - **No API Quotas**: Perform millions of route calculations without paying for cloud tokens.

→ Next: [Deep Learning Architectures](deep-learning-spatial.md)
