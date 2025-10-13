# Sample Network Datasets

This directory contains sample network datasets in various formats for testing and examples.

## Datasets

### 1. Karate Club Network
- **Files:** `karate_club.graphml`, `karate_club.gml`, `karate_club.csv`
- **Description:** Zachary's Karate Club social network
- **Nodes:** 34 | **Edges:** 78
- **Type:** Undirected, unweighted
- **Use case:** Community detection, social network analysis

### 2. Small Social Network
- **Files:** `social_network.graphml`, `social_network.csv`
- **Description:** Random social network (Erdős-Rényi)
- **Nodes:** 50 | **Edges:** ~125
- **Type:** Undirected, unweighted
- **Use case:** Social network analysis, influence propagation

### 3. Citation Network
- **Files:** `citation_network.graphml`, `citation_network.csv`
- **Description:** Paper citation network (preferential attachment)
- **Nodes:** 100 | **Edges:** ~290
- **Type:** Directed, unweighted
- **Use case:** PageRank, HITS, authority analysis

### 4. Protein Interaction Network
- **Files:** `protein_interactions.graphml`, `protein_interactions.csv`
- **Description:** Protein-protein interaction network
- **Nodes:** 80 | **Edges:** ~160
- **Type:** Undirected, unweighted
- **Use case:** Biological network analysis, module detection

### 5. Transportation Network
- **Files:** `road_network.graphml`, `road_network.csv`
- **Description:** City road network (grid structure)
- **Nodes:** 100 | **Edges:** 180
- **Type:** Undirected, unweighted
- **Use case:** Shortest path, critical infrastructure

### 6. Facebook Ego Network
- **Files:** `facebook_ego.edgelist`, `facebook_ego.csv`
- **Description:** Social network with community structure
- **Nodes:** 50 | **Edges:** ~125
- **Type:** Undirected, unweighted
- **Use case:** Community detection, ego network analysis

### 7. Small-World Network
- **Files:** `small_world.graphml`
- **Description:** Watts-Strogatz small-world network
- **Nodes:** 100 | **Edges:** 300
- **Type:** Undirected, unweighted
- **Use case:** Small-world properties, clustering

### 8. Scale-Free Network
- **Files:** `scale_free.graphml`
- **Description:** Barabási-Albert scale-free network
- **Nodes:** 100 | **Edges:** ~290
- **Type:** Undirected, unweighted
- **Use case:** Hub analysis, degree distribution

## File Formats

- **GraphML** (`.graphml`): XML-based format, preserves all graph properties
- **GML** (`.gml`): Graph Modeling Language
- **CSV** (`.csv`): Edge list with source and target columns
- **Edge List** (`.edgelist`): Simple text format with space-separated edges

## Usage

### Python
```python
from data_loader import DataLoader

# Load GraphML
G = DataLoader.load_graph('data/sample_networks/karate_club.graphml')

# Load CSV
G = DataLoader.load_from_csv('data/sample_networks/karate_club.csv')
```

### Julia
```julia
using Graphs

# Load edge list
g = loadgraph("data/sample_networks/karate_club.edgelist", "edgelist")
```

## Generating New Samples

Run the data generation script:
```bash
python data/generate_sample_data.py
```
