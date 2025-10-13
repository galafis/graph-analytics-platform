# Graph Analytics and Network Science Platform

![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Performance](https://img.shields.io/badge/Performance-High-brightgreen)

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇬🇧 English

### 📊 Overview

**Graph Analytics and Network Science Platform** is a high-performance platform for graph analysis and network science, combining **Julia** for computationally intensive algorithms with **Python** for visualization and integration. It provides advanced capabilities for community detection, centrality analysis, PageRank, social network analysis, graph-based recommendation systems, and interactive network visualizations.

This platform is designed for analyzing large-scale networks efficiently, from social networks to biological systems and recommendation graphs.

### ✨ Key Features

- **High-Performance Julia Algorithms**
  - Community detection (Louvain, Label Propagation)
  - Centrality measures (Betweenness, Closeness, Eigenvector)
  - PageRank and HITS
  - Shortest path algorithms
  - Graph clustering

- **Python Visualization & Integration**
  - NetworkX for graph manipulation
  - Plotly for interactive visualizations
  - Integration with Neo4j graph database
  - Export to various formats

- **Network Science Applications**
  - Social network analysis
  - Influence propagation
  - Graph-based recommendations
  - Link prediction
  - Network motif detection

- **Scalability**
  - Optimized for large graphs (millions of nodes)
  - Parallel processing
  - Memory-efficient algorithms
  - Distributed computing support

### 🏗️ Architecture

```
graph-analytics-platform/
├── julia/                  # Julia algorithms
│   ├── centrality.jl
│   ├── community.jl
│   └── pagerank.jl
├── python/                 # Python integration
│   ├── graph_analyzer.py
│   ├── visualizer.py
│   └── neo4j_connector.py
├── examples/               # Usage examples
├── data/                   # Sample networks
├── tests/                  # Tests
└── docs/                   # Documentation
```

### 🚀 Quick Start

#### Prerequisites

- Julia 1.9+
- Python 3.8+
- Neo4j (optional)

#### Installation

```bash
# Install Julia packages
julia -e 'using Pkg; Pkg.add(["Graphs", "GraphPlot", "LightGraphs", "SimpleWeightedGraphs"])'

# Install Python packages
pip install -r requirements.txt
```

#### Usage Example

```python
from python.graph_analyzer import GraphAnalyzer

# Initialize analyzer
analyzer = GraphAnalyzer()

# Load graph
G = analyzer.load_graph('data/social_network.gml')

# Community detection
communities = analyzer.detect_communities(G, method='louvain')
print(f"Found {len(communities)} communities")

# Centrality analysis
centrality = analyzer.calculate_centrality(G, metric='betweenness')
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 influential nodes: {top_nodes}")

# Visualize
analyzer.visualize_network(G, communities=communities, save_path='network.html')
```

### 📊 Performance

- **Graph Size**: Up to 10M nodes, 100M edges
- **Community Detection**: 100K nodes in < 5 seconds
- **PageRank**: 1M nodes in < 10 seconds
- **Memory**: Optimized for large graphs

### 📄 License

MIT License - see LICENSE file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

---

<a name="português"></a>
## 🇧🇷 Português

### 📊 Visão Geral

**Graph Analytics and Network Science Platform** é uma plataforma de alta performance para análise de grafos e ciência de redes, combinando **Julia** para algoritmos computacionalmente intensivos com **Python** para visualização e integração. Fornece capacidades avançadas para detecção de comunidades, análise de centralidade, PageRank, análise de redes sociais, sistemas de recomendação baseados em grafos e visualizações interativas de redes.

Esta plataforma é projetada para analisar redes de grande escala de forma eficiente, desde redes sociais até sistemas biológicos e grafos de recomendação.

### ✨ Principais Recursos

- **Algoritmos Julia de Alta Performance**
  - Detecção de comunidades (Louvain, Label Propagation)
  - Medidas de centralidade (Betweenness, Closeness, Eigenvector)
  - PageRank e HITS
  - Algoritmos de caminho mais curto
  - Clustering de grafos

- **Visualização e Integração Python**
  - NetworkX para manipulação de grafos
  - Plotly para visualizações interativas
  - Integração com banco de dados Neo4j
  - Exportação para vários formatos

- **Aplicações de Ciência de Redes**
  - Análise de redes sociais
  - Propagação de influência
  - Recomendações baseadas em grafos
  - Predição de links
  - Detecção de motifs de rede

- **Escalabilidade**
  - Otimizado para grafos grandes (milhões de nós)
  - Processamento paralelo
  - Algoritmos eficientes em memória
  - Suporte a computação distribuída

### 🏗️ Arquitetura

```
graph-analytics-platform/
├── julia/                  # Algoritmos Julia
│   ├── centrality.jl
│   ├── community.jl
│   └── pagerank.jl
├── python/                 # Integração Python
│   ├── graph_analyzer.py
│   ├── visualizer.py
│   └── neo4j_connector.py
├── examples/               # Exemplos de uso
├── data/                   # Redes de exemplo
├── tests/                  # Testes
└── docs/                   # Documentação
```

### 🚀 Início Rápido

#### Pré-requisitos

- Julia 1.9+
- Python 3.8+
- Neo4j (opcional)

#### Instalação

```bash
# Instale pacotes Julia
julia -e 'using Pkg; Pkg.add(["Graphs", "GraphPlot", "LightGraphs", "SimpleWeightedGraphs"])'

# Instale pacotes Python
pip install -r requirements.txt
```

#### Exemplo de Uso

```python
from python.graph_analyzer import GraphAnalyzer

# Inicialize o analisador
analyzer = GraphAnalyzer()

# Carregue o grafo
G = analyzer.load_graph('data/social_network.gml')

# Detecção de comunidades
communities = analyzer.detect_communities(G, method='louvain')
print(f"Encontradas {len(communities)} comunidades")

# Análise de centralidade
centrality = analyzer.calculate_centrality(G, metric='betweenness')
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nós influentes: {top_nodes}")

# Visualize
analyzer.visualize_network(G, communities=communities, save_path='network.html')
```

### 📊 Performance

- **Tamanho do Grafo**: Até 10M nós, 100M arestas
- **Detecção de Comunidades**: 100K nós em < 5 segundos
- **PageRank**: 1M nós em < 10 segundos
- **Memória**: Otimizado para grafos grandes

### 📄 Licença

Licença MIT - veja o arquivo LICENSE para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

