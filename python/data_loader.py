"""
Data Loader Module

Utilities for loading and saving graph data in various formats.

Author: Gabriel Demetrios Lafis
"""

import networkx as nx
import pandas as pd
from typing import Optional, Dict, List, Tuple
import os


class DataLoader:
    """
    Data loader for various graph formats.
    """
    
    @staticmethod
    def load_graph(path: str, format: Optional[str] = None) -> nx.Graph:
        """
        Load graph from file.
        
        Parameters
        ----------
        path : str
            Path to file
        format : str, optional
            File format (auto-detected if None)
            
        Returns
        -------
        G : nx.Graph
            Loaded graph
        """
        if format is None:
            # Auto-detect format from extension
            ext = os.path.splitext(path)[1].lower()
            format_map = {
                '.gml': 'gml',
                '.graphml': 'graphml',
                '.edgelist': 'edgelist',
                '.adjlist': 'adjlist',
                '.gexf': 'gexf',
                '.gpickle': 'gpickle',
                '.csv': 'csv',
                '.txt': 'edgelist'
            }
            format = format_map.get(ext, 'edgelist')
        
        if format == 'gml':
            return nx.read_gml(path)
        elif format == 'graphml':
            return nx.read_graphml(path)
        elif format == 'edgelist':
            return nx.read_edgelist(path)
        elif format == 'adjlist':
            return nx.read_adjlist(path)
        elif format == 'gexf':
            return nx.read_gexf(path)
        elif format == 'gpickle':
            return nx.read_gpickle(path)
        elif format == 'csv':
            return DataLoader.load_from_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def save_graph(G: nx.Graph, path: str, format: Optional[str] = None):
        """
        Save graph to file.
        
        Parameters
        ----------
        G : nx.Graph
            Graph to save
        path : str
            Output path
        format : str, optional
            File format (auto-detected if None)
        """
        if format is None:
            ext = os.path.splitext(path)[1].lower()
            format_map = {
                '.gml': 'gml',
                '.graphml': 'graphml',
                '.edgelist': 'edgelist',
                '.adjlist': 'adjlist',
                '.gexf': 'gexf',
                '.gpickle': 'gpickle'
            }
            format = format_map.get(ext, 'edgelist')
        
        if format == 'gml':
            nx.write_gml(G, path)
        elif format == 'graphml':
            nx.write_graphml(G, path)
        elif format == 'edgelist':
            nx.write_edgelist(G, path)
        elif format == 'adjlist':
            nx.write_adjlist(G, path)
        elif format == 'gexf':
            nx.write_gexf(G, path)
        elif format == 'gpickle':
            nx.write_gpickle(G, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_from_csv(path: str, source_col: str = 'source', target_col: str = 'target') -> nx.Graph:
        """
        Load graph from CSV file.
        
        Parameters
        ----------
        path : str
            Path to CSV file
        source_col : str
            Name of source column
        target_col : str
            Name of target column
            
        Returns
        -------
        G : nx.Graph
            Loaded graph
        """
        df = pd.read_csv(path)
        G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
        return G
    
    @staticmethod
    def save_to_csv(G: nx.Graph, path: str):
        """
        Save graph to CSV file.
        
        Parameters
        ----------
        G : nx.Graph
            Graph to save
        path : str
            Output path
        """
        edges = list(G.edges())
        df = pd.DataFrame(edges, columns=['source', 'target'])
        df.to_csv(path, index=False)
    
    @staticmethod
    def load_weighted_graph(path: str, weight_col: str = 'weight') -> nx.Graph:
        """
        Load weighted graph from CSV.
        
        Parameters
        ----------
        path : str
            Path to CSV file
        weight_col : str
            Name of weight column
            
        Returns
        -------
        G : nx.Graph
            Weighted graph
        """
        df = pd.read_csv(path)
        G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=weight_col)
        return G
    
    @staticmethod
    def graph_to_dataframe(G: nx.Graph) -> pd.DataFrame:
        """
        Convert graph to pandas DataFrame.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        df : pd.DataFrame
            Edge list as DataFrame
        """
        edges = list(G.edges(data=True))
        if edges and len(edges[0]) == 3:
            # Has edge attributes
            data = []
            for u, v, attr in edges:
                row = {'source': u, 'target': v}
                row.update(attr)
                data.append(row)
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(edges, columns=['source', 'target'])
    
    @staticmethod
    def load_karate_club() -> nx.Graph:
        """
        Load Zachary's Karate Club graph.
        
        Returns
        -------
        G : nx.Graph
            Karate club graph
        """
        return nx.karate_club_graph()
    
    @staticmethod
    def load_sample_graph(name: str) -> nx.Graph:
        """
        Load sample graph by name.
        
        Parameters
        ----------
        name : str
            Graph name ('karate_club', 'davis_southern', 'florentine_families', etc.)
            
        Returns
        -------
        G : nx.Graph
            Sample graph
        """
        if name == 'karate_club':
            return nx.karate_club_graph()
        elif name == 'davis_southern':
            return nx.davis_southern_women_graph()
        elif name == 'florentine_families':
            return nx.florentine_families_graph()
        elif name == 'les_miserables':
            return nx.les_miserables_graph()
        else:
            raise ValueError(f"Unknown sample graph: {name}")
    
    @staticmethod
    def generate_sample_data(n_nodes: int, edge_prob: float, save_dir: str = 'data/sample_networks'):
        """
        Generate and save sample graph data.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes
        edge_prob : float
            Edge probability
        save_dir : str
            Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate random graph
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        
        # Save in different formats
        DataLoader.save_graph(G, os.path.join(save_dir, 'sample_graph.gml'), 'gml')
        DataLoader.save_graph(G, os.path.join(save_dir, 'sample_graph.graphml'), 'graphml')
        DataLoader.save_graph(G, os.path.join(save_dir, 'sample_graph.edgelist'), 'edgelist')
        DataLoader.save_to_csv(G, os.path.join(save_dir, 'sample_graph.csv'))
        
        print(f"Sample data generated in {save_dir}")
        print(f"Graph: {n_nodes} nodes, {G.number_of_edges()} edges")

