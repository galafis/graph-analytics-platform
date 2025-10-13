# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-13

### Added

#### Core Modules
- **Julia Modules**
  - `centrality.jl` - Degree, betweenness, closeness, eigenvector centrality algorithms
  - `pagerank.jl` - PageRank, HITS, personalized PageRank, k-shell decomposition
  - `community.jl` - Label propagation, modularity optimization, connected components
  - `shortest_paths.jl` - Dijkstra, Bellman-Ford, Floyd-Warshall, A* algorithms
  - `link_prediction.jl` - Common neighbors, Jaccard, Adamic-Adar, preferential attachment
  - `utils.jl` - Utility functions for graph manipulation and analysis

- **Python Modules**
  - `graph_analyzer.py` - High-level graph analysis interface
  - `julia_bridge.py` - Julia-Python integration layer
  - `visualization.py` - Network visualization tools (Matplotlib, Plotly)
  - `data_loader.py` - Graph data loading and saving utilities
  - `metrics.py` - Comprehensive network metrics calculation

#### Features
- **Centrality Analysis**
  - Multiple centrality measures (degree, betweenness, closeness, eigenvector, PageRank)
  - Batch centrality calculation
  - Top influencers identification

- **Community Detection**
  - Louvain algorithm
  - Label propagation
  - Greedy modularity optimization
  - Modularity calculation

- **Link Prediction**
  - Common neighbors score
  - Jaccard coefficient
  - Adamic-Adar index
  - Preferential attachment
  - Resource allocation index

- **Path Analysis**
  - Dijkstra's algorithm
  - Bellman-Ford algorithm (handles negative weights)
  - Floyd-Warshall (all-pairs shortest paths)
  - A* algorithm with heuristics

- **Visualization**
  - Interactive network visualization (Plotly)
  - Static network plots (Matplotlib)
  - Degree distribution plots
  - Community structure visualization
  - Centrality comparison plots

#### Testing
- Python integration tests with 15 test cases
- Julia unit tests for centrality algorithms
- Julia unit tests for community detection
- Julia unit tests for PageRank algorithms
- All tests passing successfully

#### Examples
- `social_network_analysis.py` - Comprehensive social network analysis
- `citation_network.jl` - Citation network and research community analysis
- `biological_network.py` - Protein-protein interaction network analysis
- `transportation_network.jl` - Road network and route optimization

#### Data
- Sample network datasets (8 different networks)
- Karate club network
- Social network
- Citation network
- Protein interaction network
- Transportation network
- Facebook ego network
- Small-world network
- Scale-free network
- Data in multiple formats (GraphML, GML, CSV, edgelist)

#### Documentation
- Comprehensive README.md with detailed examples
- API documentation in code (docstrings)
- Contributing guidelines (CONTRIBUTING.md)
- Sample data README
- Installation instructions
- Usage examples for both Python and Julia

### Changed
- Enhanced `graph_analyzer.py` with additional methods
- Improved error handling in all modules
- Optimized centrality calculations

### Fixed
- Type consistency in link prediction scores
- Edge cases in clustering coefficient calculation
- Handling of disconnected graphs in path metrics

## [0.1.0] - 2025-10-12

### Added
- Initial repository structure
- Basic Julia centrality module
- Basic Python graph analyzer
- Basic community detection
- Basic PageRank implementation
- Initial README.md
- LICENSE file

---

## Legend

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

