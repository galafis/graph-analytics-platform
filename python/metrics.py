"""
Network Metrics Module

Comprehensive network metrics and analysis functions.

Author: Gabriel Demetrios Lafis
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class NetworkMetrics:
    """
    Calculate comprehensive network metrics.
    """
    
    @staticmethod
    def basic_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        Calculate basic network metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            Basic metrics
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        return {
            'num_nodes': n,
            'num_edges': m,
            'density': nx.density(G),
            'avg_degree': np.mean(degree_values) if degree_values else 0,
            'max_degree': max(degree_values) if degree_values else 0,
            'min_degree': min(degree_values) if degree_values else 0,
            'degree_variance': np.var(degree_values) if degree_values else 0
        }
    
    @staticmethod
    def clustering_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        Calculate clustering-related metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            Clustering metrics
        """
        return {
            'avg_clustering': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
            'triangles': sum(nx.triangles(G).values()) // 3
        }
    
    @staticmethod
    def connectivity_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        Calculate connectivity metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            Connectivity metrics
        """
        metrics = {
            'is_connected': nx.is_connected(G),
            'num_components': nx.number_connected_components(G)
        }
        
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['radius'] = nx.radius(G)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            # Calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['radius'] = nx.radius(subgraph)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)
            metrics['largest_component_size'] = len(largest_cc)
            metrics['largest_component_fraction'] = len(largest_cc) / G.number_of_nodes()
        
        return metrics
    
    @staticmethod
    def centrality_metrics(G: nx.Graph) -> Dict[str, Dict[int, float]]:
        """
        Calculate multiple centrality measures.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        centralities : dict
            Dictionary of centrality measures
        """
        return {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
            'pagerank': nx.pagerank(G)
        }
    
    @staticmethod
    def degree_distribution(G: nx.Graph) -> Dict[int, int]:
        """
        Calculate degree distribution.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        distribution : dict
            Degree distribution
        """
        degrees = [G.degree(n) for n in G.nodes()]
        return dict(Counter(degrees))
    
    @staticmethod
    def assortativity_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        Calculate assortativity metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            Assortativity metrics
        """
        return {
            'degree_assortativity': nx.degree_assortativity_coefficient(G),
        }
    
    @staticmethod
    def small_world_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        Calculate small-world metrics (Watts-Strogatz coefficients).
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            Small-world metrics
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        avg_degree = 2 * m / n if n > 0 else 0
        
        # Calculate clustering coefficient
        C = nx.average_clustering(G)
        
        # Calculate average path length
        if nx.is_connected(G):
            L = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            L = nx.average_shortest_path_length(subgraph)
        
        # Generate random graph for comparison
        G_random = nx.erdos_renyi_graph(n, 2 * m / (n * (n - 1)))
        C_random = nx.average_clustering(G_random)
        
        if nx.is_connected(G_random):
            L_random = nx.average_shortest_path_length(G_random)
        else:
            largest_cc = max(nx.connected_components(G_random), key=len)
            subgraph = G_random.subgraph(largest_cc)
            L_random = nx.average_shortest_path_length(subgraph)
        
        # Small-world coefficient
        sigma = (C / C_random) / (L / L_random) if C_random > 0 and L_random > 0 else 0
        
        return {
            'clustering_coefficient': C,
            'avg_path_length': L,
            'clustering_random': C_random,
            'path_length_random': L_random,
            'small_world_coefficient': sigma
        }
    
    @staticmethod
    def all_metrics(G: nx.Graph) -> Dict:
        """
        Calculate all available metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        metrics : dict
            All metrics
        """
        return {
            'basic': NetworkMetrics.basic_metrics(G),
            'clustering': NetworkMetrics.clustering_metrics(G),
            'connectivity': NetworkMetrics.connectivity_metrics(G),
            'assortativity': NetworkMetrics.assortativity_metrics(G)
        }
    
    @staticmethod
    def compare_graphs(G1: nx.Graph, G2: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """
        Compare metrics between two graphs.
        
        Parameters
        ----------
        G1 : nx.Graph
            First graph
        G2 : nx.Graph
            Second graph
            
        Returns
        -------
        comparison : dict
            Comparison of metrics
        """
        metrics1 = NetworkMetrics.basic_metrics(G1)
        metrics2 = NetworkMetrics.basic_metrics(G2)
        
        comparison = {}
        for key in metrics1:
            comparison[key] = (metrics1[key], metrics2[key])
        
        return comparison
    
    @staticmethod
    def print_summary(G: nx.Graph):
        """
        Print summary of graph metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        """
        print("\n" + "="*50)
        print("NETWORK SUMMARY")
        print("="*50)
        
        basic = NetworkMetrics.basic_metrics(G)
        print("\nBasic Metrics:")
        for key, value in basic.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        clustering = NetworkMetrics.clustering_metrics(G)
        print("\nClustering Metrics:")
        for key, value in clustering.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        connectivity = NetworkMetrics.connectivity_metrics(G)
        print("\nConnectivity Metrics:")
        for key, value in connectivity.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n" + "="*50)

