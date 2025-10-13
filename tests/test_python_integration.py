"""
Python Integration Tests

Tests for Python graph analysis functionality.

Author: Gabriel Demetrios Lafis
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import networkx as nx
from graph_analyzer import GraphAnalyzer


class TestGraphAnalyzer(unittest.TestCase):
    """Test GraphAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GraphAnalyzer()
    
    def test_create_random_graph(self):
        """Test random graph generation."""
        G = self.analyzer.create_random_graph(10, 0.3, 'erdos_renyi')
        self.assertEqual(G.number_of_nodes(), 10)
        self.assertIsInstance(G, nx.Graph)
    
    def test_centrality_calculations(self):
        """Test centrality calculations."""
        G = nx.karate_club_graph()
        self.analyzer.graph = G
        
        # Test degree centrality
        dc = self.analyzer.calculate_centrality(G, 'degree')
        self.assertEqual(len(dc), G.number_of_nodes())
        self.assertTrue(all(0 <= v <= 1 for v in dc.values()))
        
        # Test betweenness centrality
        bc = self.analyzer.calculate_centrality(G, 'betweenness')
        self.assertEqual(len(bc), G.number_of_nodes())
        self.assertTrue(all(v >= 0 for v in bc.values()))
    
    def test_community_detection(self):
        """Test community detection."""
        G = nx.karate_club_graph()
        
        # Test Louvain
        communities = self.analyzer.detect_communities(G, 'louvain')
        self.assertIsInstance(communities, list)
        self.assertTrue(len(communities) > 0)
        
        # Test label propagation
        communities = self.analyzer.detect_communities(G, 'label_propagation')
        self.assertIsInstance(communities, list)
        self.assertTrue(len(communities) > 0)
    
    def test_pagerank(self):
        """Test PageRank calculation."""
        G = nx.karate_club_graph()
        pr = self.analyzer.calculate_pagerank(G)
        
        self.assertEqual(len(pr), G.number_of_nodes())
        self.assertTrue(all(v >= 0 for v in pr.values()))
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=5)
    
    def test_shortest_path(self):
        """Test shortest path finding."""
        G = nx.path_graph(5)
        self.analyzer.graph = G
        
        path = self.analyzer.find_shortest_path(0, 4, G)
        self.assertEqual(len(path), 5)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 4)
    
    def test_clustering_coefficient(self):
        """Test clustering coefficient."""
        G = nx.complete_graph(5)
        cc = self.analyzer.calculate_clustering_coefficient(G)
        self.assertAlmostEqual(cc, 1.0, places=5)
        
        # Test on empty graph
        G_empty = nx.Graph()
        G_empty.add_nodes_from(range(3))
        cc_empty = self.analyzer.calculate_clustering_coefficient(G_empty)
        self.assertEqual(cc_empty, 0.0)
    
    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        G = nx.karate_club_graph()
        stats = self.analyzer.get_graph_statistics(G)
        
        self.assertEqual(stats['num_nodes'], G.number_of_nodes())
        self.assertEqual(stats['num_edges'], G.number_of_edges())
        self.assertIn('density', stats)
        self.assertIn('avg_degree', stats)
        self.assertIn('avg_clustering', stats)
    
    def test_network_metrics(self):
        """Test comprehensive network metrics."""
        G = nx.karate_club_graph()
        metrics = self.analyzer.calculate_network_metrics(G)
        
        self.assertIn('num_nodes', metrics)
        self.assertIn('num_edges', metrics)
        self.assertIn('density', metrics)
        self.assertIn('avg_clustering', metrics)
        self.assertTrue(metrics['num_nodes'] > 0)
        self.assertTrue(metrics['num_edges'] > 0)
    
    def test_link_prediction(self):
        """Test link prediction methods."""
        G = nx.karate_club_graph()
        self.analyzer.graph = G
        
        # Test common neighbors
        score = self.analyzer.common_neighbors_score(0, 1, G)
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)
        
        # Test Jaccard coefficient
        score = self.analyzer.jaccard_coefficient(0, 1, G)
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
        
        # Test Adamic-Adar
        score = self.analyzer.adamic_adar_score(0, 1, G)
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)
        
        # Test preferential attachment
        score = self.analyzer.preferential_attachment_score(0, 1, G)
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)
    
    def test_all_centralities(self):
        """Test calculate all centralities."""
        G = nx.karate_club_graph()
        centralities = self.analyzer.calculate_all_centralities(G)
        
        self.assertIn('degree', centralities)
        self.assertIn('betweenness', centralities)
        self.assertIn('closeness', centralities)
        self.assertIn('eigenvector', centralities)
        self.assertIn('pagerank', centralities)
        
        for cent_type, cent_values in centralities.items():
            self.assertEqual(len(cent_values), G.number_of_nodes())
    
    def test_top_influencers(self):
        """Test top influencers identification."""
        G = nx.karate_club_graph()
        influencers = self.analyzer.top_influencers(method='pagerank', top_k=5, G=G)
        
        self.assertEqual(len(influencers), 5)
        self.assertIsInstance(influencers[0], tuple)
        self.assertEqual(len(influencers[0]), 2)
        
        # Check that scores are in descending order
        scores = [score for _, score in influencers]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_identify_bridges(self):
        """Test bridge identification."""
        # Create graph with known bridge
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        
        bridges = self.analyzer.identify_bridges(G)
        self.assertIsInstance(bridges, list)
        # All edges are bridges in a path graph
        self.assertEqual(len(bridges), 5)


class TestGraphCreation(unittest.TestCase):
    """Test graph creation methods."""
    
    def test_erdos_renyi(self):
        """Test Erdős-Rényi graph creation."""
        analyzer = GraphAnalyzer()
        G = analyzer.create_random_graph(20, 0.2, 'erdos_renyi')
        self.assertEqual(G.number_of_nodes(), 20)
    
    def test_barabasi_albert(self):
        """Test Barabási-Albert graph creation."""
        analyzer = GraphAnalyzer()
        G = analyzer.create_random_graph(20, 0.1, 'barabasi_albert')
        self.assertEqual(G.number_of_nodes(), 20)
    
    def test_watts_strogatz(self):
        """Test Watts-Strogatz graph creation."""
        analyzer = GraphAnalyzer()
        G = analyzer.create_random_graph(20, 0.1, 'watts_strogatz')
        self.assertEqual(G.number_of_nodes(), 20)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Python Integration Tests")
    print("="*60 + "\n")
    run_tests()
    print("\n✓ All Python tests completed!")
