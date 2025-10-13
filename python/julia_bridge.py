"""
Julia-Python Bridge Module

Provides seamless integration between Julia and Python for
high-performance graph analytics.

Author: Gabriel Demetrios Lafis
"""

try:
    from julia import Main
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    import warnings
    warnings.warn("Julia integration not available. Install PyJulia: pip install julia")

import networkx as nx
from typing import Dict, List, Optional, Tuple


class JuliaBridge:
    """
    Bridge between Python NetworkX and Julia Graphs.jl.
    
    Provides high-performance implementations of graph algorithms
    using Julia's JIT compilation.
    """
    
    def __init__(self):
        """Initialize Julia bridge."""
        self.julia_available = JULIA_AVAILABLE
        
        if self.julia_available:
            try:
                # Load Julia modules
                Main.eval('using Graphs')
                Main.eval('using LinearAlgebra')
                Main.eval('using SparseArrays')
                Main.eval('using DataStructures')
                
                # Load custom Julia modules
                Main.eval('include("julia/centrality.jl")')
                Main.eval('include("julia/pagerank.jl")')
                Main.eval('include("julia/community.jl")')
                Main.eval('include("julia/shortest_paths.jl")')
                Main.eval('include("julia/link_prediction.jl")')
                Main.eval('include("julia/utils.jl")')
                
            except Exception as e:
                self.julia_available = False
                import warnings
                warnings.warn(f"Failed to load Julia modules: {e}")
    
    def nx_to_julia(self, G: nx.Graph):
        """
        Convert NetworkX graph to Julia SimpleGraph.
        
        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
            
        Returns
        -------
        g : Julia SimpleGraph
            Julia graph object
        """
        if not self.julia_available:
            raise RuntimeError("Julia integration not available")
        
        n = G.number_of_nodes()
        
        # Create Julia graph
        Main.eval(f'g = SimpleGraph({n})')
        
        # Add edges
        for u, v in G.edges():
            Main.eval(f'add_edge!(g, {u}, {v})')
        
        return Main.g
    
    def julia_to_nx(self, g) -> nx.Graph:
        """
        Convert Julia graph to NetworkX graph.
        
        Parameters
        ----------
        g : Julia SimpleGraph
            Julia graph
            
        Returns
        -------
        G : nx.Graph
            NetworkX graph
        """
        if not self.julia_available:
            raise RuntimeError("Julia integration not available")
        
        # Get number of vertices
        n = int(Main.eval('nv(g)'))
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(1, n + 1))
        
        # Get edges
        Main.eval('edges_list = [(src(e), dst(e)) for e in edges(g)]')
        edges_list = Main.edges_list
        
        for edge in edges_list:
            G.add_edge(edge[0], edge[1])
        
        return G
    
    def betweenness_centrality(self, G: nx.Graph) -> Dict[int, float]:
        """
        Calculate betweenness centrality using Julia.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        centrality : dict
            Betweenness centrality scores
        """
        if not self.julia_available:
            return nx.betweenness_centrality(G)
        
        g = self.nx_to_julia(G)
        result = Main.eval('betweenness_centrality(g)')
        return dict(result)
    
    def pagerank(
        self,
        G: nx.Graph,
        alpha: float = 0.85,
        max_iter: int = 100
    ) -> Dict[int, float]:
        """
        Calculate PageRank using Julia.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        alpha : float
            Damping factor
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        pagerank : dict
            PageRank scores
        """
        if not self.julia_available:
            return nx.pagerank(G, alpha=alpha, max_iter=max_iter)
        
        g = self.nx_to_julia(G)
        Main.alpha = alpha
        Main.max_iter = max_iter
        result = Main.eval('pagerank(g, alpha=alpha, max_iter=max_iter)')
        return dict(result)
    
    def label_propagation(self, G: nx.Graph) -> Dict[int, int]:
        """
        Detect communities using label propagation in Julia.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        communities : dict
            Community assignments
        """
        if not self.julia_available:
            # Fallback to NetworkX
            comms = nx.community.label_propagation_communities(G)
            result = {}
            for i, comm in enumerate(comms):
                for node in comm:
                    result[node] = i
            return result
        
        g = self.nx_to_julia(G)
        result = Main.eval('label_propagation(g)')
        return dict(result)
    
    def dijkstra(self, G: nx.Graph, source: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        Calculate shortest paths using Dijkstra's algorithm in Julia.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        source : int
            Source node
            
        Returns
        -------
        distances : dict
            Distances from source
        predecessors : dict
            Predecessor nodes
        """
        if not self.julia_available:
            distances = nx.single_source_dijkstra_path_length(G, source)
            paths = nx.single_source_dijkstra_path(G, source)
            predecessors = {}
            for target, path in paths.items():
                if len(path) > 1:
                    predecessors[target] = path[-2]
            return distances, predecessors
        
        g = self.nx_to_julia(G)
        Main.source = source
        Main.eval('dist, pred = dijkstra(g, source)')
        
        distances = dict(Main.dist)
        predecessors = dict(Main.pred)
        
        return distances, predecessors
    
    def common_neighbors(self, G: nx.Graph, u: int, v: int) -> float:
        """
        Calculate common neighbors score using Julia.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        u : int
            First node
        v : int
            Second node
            
        Returns
        -------
        score : float
            Common neighbors score
        """
        if not self.julia_available:
            return float(len(list(nx.common_neighbors(G, u, v))))
        
        g = self.nx_to_julia(G)
        Main.u = u
        Main.v = v
        return float(Main.eval('common_neighbors(g, u, v)'))

