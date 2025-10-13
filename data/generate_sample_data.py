"""
Generate Sample Network Data

Creates sample network datasets in various formats for testing and examples.

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import networkx as nx
from data_loader import DataLoader


def generate_all_samples():
    """Generate all sample network data."""
    
    print("\n" + "="*60)
    print("GENERATING SAMPLE NETWORK DATA")
    print("="*60 + "\n")
    
    output_dir = 'data/sample_networks'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Karate Club Network
    print("1. Generating Karate Club Network...")
    G_karate = nx.karate_club_graph()
    DataLoader.save_graph(G_karate, os.path.join(output_dir, 'karate_club.graphml'), 'graphml')
    DataLoader.save_graph(G_karate, os.path.join(output_dir, 'karate_club.gml'), 'gml')
    DataLoader.save_to_csv(G_karate, os.path.join(output_dir, 'karate_club.csv'))
    print(f"   Saved: {G_karate.number_of_nodes()} nodes, {G_karate.number_of_edges()} edges")
    
    # 2. Small social network
    print("\n2. Generating Small Social Network...")
    G_social = nx.erdos_renyi_graph(50, 0.1, seed=42)
    DataLoader.save_graph(G_social, os.path.join(output_dir, 'social_network.graphml'), 'graphml')
    DataLoader.save_to_csv(G_social, os.path.join(output_dir, 'social_network.csv'))
    print(f"   Saved: {G_social.number_of_nodes()} nodes, {G_social.number_of_edges()} edges")
    
    # 3. Citation network
    print("\n3. Generating Citation Network...")
    G_citation = nx.barabasi_albert_graph(100, 3, seed=42)
    # Convert to directed for citations
    G_citation_dir = nx.DiGraph(G_citation)
    DataLoader.save_graph(G_citation_dir, os.path.join(output_dir, 'citation_network.graphml'), 'graphml')
    DataLoader.save_to_csv(G_citation_dir, os.path.join(output_dir, 'citation_network.csv'))
    print(f"   Saved: {G_citation_dir.number_of_nodes()} nodes, {G_citation_dir.number_of_edges()} edges")
    
    # 4. Protein interaction network
    print("\n4. Generating Protein Interaction Network...")
    G_protein = nx.watts_strogatz_graph(80, 4, 0.3, seed=42)
    DataLoader.save_graph(G_protein, os.path.join(output_dir, 'protein_interactions.graphml'), 'graphml')
    DataLoader.save_to_csv(G_protein, os.path.join(output_dir, 'protein_interactions.csv'))
    print(f"   Saved: {G_protein.number_of_nodes()} nodes, {G_protein.number_of_edges()} edges")
    
    # 5. Transportation network (grid)
    print("\n5. Generating Transportation Network...")
    G_transport = nx.grid_2d_graph(10, 10)
    # Convert to simple integer nodes
    G_transport_simple = nx.Graph()
    node_map = {node: i for i, node in enumerate(G_transport.nodes())}
    G_transport_simple.add_nodes_from(range(len(node_map)))
    for u, v in G_transport.edges():
        G_transport_simple.add_edge(node_map[u], node_map[v])
    
    DataLoader.save_graph(G_transport_simple, os.path.join(output_dir, 'road_network.graphml'), 'graphml')
    DataLoader.save_to_csv(G_transport_simple, os.path.join(output_dir, 'road_network.csv'))
    print(f"   Saved: {G_transport_simple.number_of_nodes()} nodes, {G_transport_simple.number_of_edges()} edges")
    
    # 6. Facebook ego network (sample)
    print("\n6. Generating Facebook Ego Network...")
    G_facebook = nx.connected_caveman_graph(5, 10)
    DataLoader.save_graph(G_facebook, os.path.join(output_dir, 'facebook_ego.edgelist'), 'edgelist')
    DataLoader.save_to_csv(G_facebook, os.path.join(output_dir, 'facebook_ego.csv'))
    print(f"   Saved: {G_facebook.number_of_nodes()} nodes, {G_facebook.number_of_edges()} edges")
    
    # 7. Small-world network
    print("\n7. Generating Small-World Network...")
    G_smallworld = nx.watts_strogatz_graph(100, 6, 0.1, seed=42)
    DataLoader.save_graph(G_smallworld, os.path.join(output_dir, 'small_world.graphml'), 'graphml')
    print(f"   Saved: {G_smallworld.number_of_nodes()} nodes, {G_smallworld.number_of_edges()} edges")
    
    # 8. Scale-free network
    print("\n8. Generating Scale-Free Network...")
    G_scalefree = nx.barabasi_albert_graph(100, 3, seed=42)
    DataLoader.save_graph(G_scalefree, os.path.join(output_dir, 'scale_free.graphml'), 'graphml')
    print(f"   Saved: {G_scalefree.number_of_nodes()} nodes, {G_scalefree.number_of_edges()} edges")
    
    # Create README
    print("\n9. Creating README for sample data...")
    readme_content = """# Sample Network Datasets

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
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("="*60)
    print(f"All sample datasets saved to: {output_dir}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    generate_all_samples()
