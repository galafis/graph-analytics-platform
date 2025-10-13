"""
Graph Analysis Module

Provides high-level interface for graph analytics using
Julia for performance-critical algorithms and Python for
visualization and integration.

Author: Gabriel Demetrios Lafis
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import community as community_louvain
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GraphAnalyzer:
    """
    High-level graph analysis interface.
    
    Combines NetworkX for graph manipulation with custom
    algorithms for advanced analytics.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize graph analyzer.
        
        Parameters
        ----------
        graph : nx.Graph, optional
            NetworkX graph to analyze. If None, use load_graph() to load one.
        """
        self.graph = graph
        
    def load_graph(self, path: str, format: str = 'gml') -> nx.Graph:
        """
        Load graph from file.
        
        Parameters
        ----------
        path : str
            Path to graph file
        format : str
            File format (gml, graphml, edgelist, etc.)
            
        Returns
        -------
        G : nx.Graph
            Loaded graph
        """
        if format == 'gml':
            self.graph = nx.read_gml(path)
        elif format == 'graphml':
            self.graph = nx.read_graphml(path)
        elif format == 'edgelist':
            self.graph = nx.read_edgelist(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.graph
    
    def create_random_graph(
        self,
        n_nodes: int,
        edge_prob: float = 0.1,
        graph_type: str = 'erdos_renyi'
    ) -> nx.Graph:
        """
        Create random graph for testing.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes
        edge_prob : float
            Edge probability
        graph_type : str
            Type of random graph
            
        Returns
        -------
        G : nx.Graph
            Generated graph
        """
        if graph_type == 'erdos_renyi':
            self.graph = nx.erdos_renyi_graph(n_nodes, edge_prob)
        elif graph_type == 'barabasi_albert':
            m = max(1, int(n_nodes * edge_prob))
            self.graph = nx.barabasi_albert_graph(n_nodes, m)
        elif graph_type == 'watts_strogatz':
            k = max(2, int(n_nodes * edge_prob))
            self.graph = nx.watts_strogatz_graph(n_nodes, k, 0.1)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        return self.graph
    
    def detect_communities(
        self,
        G: Optional[nx.Graph] = None,
        method: str = 'louvain'
    ) -> List[Set[int]]:
        """
        Detect communities in graph.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze (uses self.graph if None)
        method : str
            Community detection method
            
        Returns
        -------
        communities : list of sets
            Detected communities
        """
        if G is None:
            G = self.graph
        
        if method == 'louvain':
            partition = community_louvain.best_partition(G)
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = set()
                communities[comm_id].add(node)
            return list(communities.values())
        
        elif method == 'label_propagation':
            communities = nx.community.label_propagation_communities(G)
            return [set(c) for c in communities]
        
        elif method == 'greedy_modularity':
            communities = nx.community.greedy_modularity_communities(G)
            return [set(c) for c in communities]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_centrality(
        self,
        G: Optional[nx.Graph] = None,
        metric: str = 'betweenness'
    ) -> Dict[int, float]:
        """
        Calculate node centrality.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
        metric : str
            Centrality metric
            
        Returns
        -------
        centrality : dict
            Centrality scores
        """
        if G is None:
            G = self.graph
        
        if metric == 'betweenness':
            return nx.betweenness_centrality(G)
        elif metric == 'closeness':
            return nx.closeness_centrality(G)
        elif metric == 'eigenvector':
            return nx.eigenvector_centrality(G, max_iter=1000)
        elif metric == 'degree':
            return nx.degree_centrality(G)
        elif metric == 'pagerank':
            return nx.pagerank(G)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def calculate_pagerank(
        self,
        G: Optional[nx.Graph] = None,
        alpha: float = 0.85,
        max_iter: int = 100
    ) -> Dict[int, float]:
        """
        Calculate PageRank scores.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
        alpha : float
            Damping parameter
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        pagerank : dict
            PageRank scores
        """
        if G is None:
            G = self.graph
        
        return nx.pagerank(G, alpha=alpha, max_iter=max_iter)
    
    def find_shortest_path(
        self,
        source: int,
        target: int,
        G: Optional[nx.Graph] = None
    ) -> List[int]:
        """
        Find shortest path between two nodes.
        
        Parameters
        ----------
        source : int
            Source node
        target : int
            Target node
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        path : list
            Shortest path
        """
        if G is None:
            G = self.graph
        
        return nx.shortest_path(G, source, target)
    
    def calculate_clustering_coefficient(
        self,
        G: Optional[nx.Graph] = None
    ) -> float:
        """
        Calculate average clustering coefficient.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        clustering : float
            Average clustering coefficient
        """
        if G is None:
            G = self.graph
        
        return nx.average_clustering(G)
    
    def visualize_network(
        self,
        G: Optional[nx.Graph] = None,
        communities: Optional[List[Set[int]]] = None,
        centrality: Optional[Dict[int, float]] = None,
        save_path: Optional[str] = None,
        title: str = "Network Visualization"
    ):
        """
        Create interactive network visualization.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to visualize
        communities : list of sets, optional
            Community assignments for coloring
        centrality : dict, optional
            Centrality scores for node sizing
        save_path : str, optional
            Path to save HTML file
        title : str
            Plot title
        """
        if G is None:
            G = self.graph
        
        # Calculate layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        # Assign colors based on communities
        if communities:
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            degree = G.degree(node)
            node_text.append(f'Node {node}<br>Degree: {degree}')
            
            # Color by community
            if communities:
                node_color.append(node_to_comm.get(node, 0))
            else:
                node_color.append(degree)
            
            # Size by centrality
            if centrality:
                node_size.append(10 + 40 * centrality.get(node, 0))
            else:
                node_size.append(10 + degree)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    title="Community" if communities else "Degree",
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
        
        return fig
    
    def get_graph_statistics(self, G: Optional[nx.Graph] = None) -> Dict:
        """
        Calculate comprehensive graph statistics.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        stats : dict
            Graph statistics
        """
        if G is None:
            G = self.graph
        
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'avg_clustering': nx.average_clustering(G),
            'num_connected_components': nx.number_connected_components(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else None,
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else None
        }



    def calculate_network_metrics(self, G: Optional[nx.Graph] = None) -> Dict[str, float]:
        """
        Calculate comprehensive network metrics.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze (uses self.graph if None)
            
        Returns
        -------
        metrics : dict
            Dictionary of network metrics including:
            - avg_degree: Average node degree
            - density: Graph density
            - avg_clustering: Average clustering coefficient
            - transitivity: Graph transitivity
            - avg_path_length: Average shortest path length (if connected)
            - diameter: Graph diameter (if connected)
            - num_nodes: Number of nodes
            - num_edges: Number of edges
        """
        if G is None:
            G = self.graph
        
        if G is None:
            raise ValueError("No graph available. Pass a graph or use load_graph()/create_random_graph() first.")
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        
        # Degree metrics
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees) if degrees else 0
        metrics['max_degree'] = max(degrees) if degrees else 0
        metrics['min_degree'] = min(degrees) if degrees else 0
        
        # Density
        metrics['density'] = nx.density(G)
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Path metrics (only for connected graphs)
        if nx.is_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['num_components'] = nx.number_connected_components(G)
        
        return metrics
    
    def common_neighbors_score(self, u: int, v: int, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate common neighbors score for link prediction.
        
        Parameters
        ----------
        u : int
            First node
        v : int
            Second node
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        score : float
            Number of common neighbors
        """
        if G is None:
            G = self.graph
        
        return float(len(list(nx.common_neighbors(G, u, v))))
    
    def jaccard_coefficient(self, u: int, v: int, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate Jaccard coefficient for link prediction.
        
        Parameters
        ----------
        u : int
            First node
        v : int
            Second node
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        score : float
            Jaccard coefficient
        """
        if G is None:
            G = self.graph
        
        preds = nx.jaccard_coefficient(G, [(u, v)])
        for _, _, score in preds:
            return score
        return 0.0
    
    def adamic_adar_score(self, u: int, v: int, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate Adamic-Adar score for link prediction.
        
        Parameters
        ----------
        u : int
            First node
        v : int
            Second node
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        score : float
            Adamic-Adar score
        """
        if G is None:
            G = self.graph
        
        preds = nx.adamic_adar_index(G, [(u, v)])
        for _, _, score in preds:
            return score
        return 0.0
    
    def preferential_attachment_score(self, u: int, v: int, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate preferential attachment score for link prediction.
        
        Parameters
        ----------
        u : int
            First node
        v : int
            Second node
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        score : float
            Product of node degrees
        """
        if G is None:
            G = self.graph
        
        preds = nx.preferential_attachment(G, [(u, v)])
        for _, _, score in preds:
            return float(score)
        return 0.0
    
    def calculate_all_centralities(self, G: Optional[nx.Graph] = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate all centrality measures.
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        centralities : dict
            Dictionary with centrality measures
        """
        if G is None:
            G = self.graph
        
        return {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
            'pagerank': nx.pagerank(G)
        }
    
    def top_influencers(
        self,
        method: str = 'pagerank',
        top_k: int = 10,
        G: Optional[nx.Graph] = None
    ) -> List[Tuple[int, float]]:
        """
        Find top influencers in the network.
        
        Parameters
        ----------
        method : str
            Centrality method to use
        top_k : int
            Number of top influencers to return
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        influencers : list of tuples
            List of (node, score) tuples
        """
        if G is None:
            G = self.graph
        
        centrality = self.calculate_centrality(G, metric=method)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def identify_bridges(self, G: Optional[nx.Graph] = None) -> List[Tuple[int, int]]:
        """
        Find bridges in the graph (edges whose removal disconnects the graph).
        
        Parameters
        ----------
        G : nx.Graph, optional
            Graph to analyze
            
        Returns
        -------
        bridges : list of tuples
            List of bridge edges
        """
        if G is None:
            G = self.graph
        
        return list(nx.bridges(G))

