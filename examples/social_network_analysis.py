"""
Social Network Analysis Example

Demonstrates comprehensive social network analysis including:
- Community detection
- Influence identification
- Link prediction
- Network visualization

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import networkx as nx
import matplotlib.pyplot as plt
from graph_analyzer import GraphAnalyzer
from visualization import visualize_network_matplotlib, visualize_communities, visualize_centrality_comparison
from metrics import NetworkMetrics

def main():
    print("\n" + "="*60)
    print("SOCIAL NETWORK ANALYSIS EXAMPLE")
    print("="*60 + "\n")
    
    # 1. Load Zachary's Karate Club Network
    print("1. Loading Zachary's Karate Club Network...")
    G = nx.karate_club_graph()
    print(f"   Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. Initialize analyzer
    analyzer = GraphAnalyzer(G)
    
    # 3. Calculate basic metrics
    print("\n2. Calculating Network Metrics...")
    metrics = NetworkMetrics.basic_metrics(G)
    print(f"   - Density: {metrics['density']:.4f}")
    print(f"   - Average degree: {metrics['avg_degree']:.2f}")
    print(f"   - Max degree: {metrics['max_degree']}")
    
    clustering = NetworkMetrics.clustering_metrics(G)
    print(f"   - Average clustering: {clustering['avg_clustering']:.4f}")
    print(f"   - Transitivity: {clustering['transitivity']:.4f}")
    print(f"   - Triangles: {clustering['triangles']}")
    
    # 4. Detect communities
    print("\n3. Detecting Communities...")
    communities_louvain = analyzer.detect_communities(method='louvain')
    print(f"   - Louvain: {len(communities_louvain)} communities detected")
    for i, comm in enumerate(communities_louvain):
        print(f"     Community {i+1}: {len(comm)} members")
    
    communities_label = analyzer.detect_communities(method='label_propagation')
    print(f"   - Label Propagation: {len(communities_label)} communities detected")
    
    # 5. Calculate centrality measures
    print("\n4. Identifying Influential Nodes...")
    centrality_results = analyzer.calculate_all_centralities()
    
    print("   Top 5 nodes by different centrality measures:")
    for measure_name, centrality in centrality_results.items():
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   {measure_name.upper()}:")
        for node, score in top_nodes:
            print(f"     Node {node}: {score:.4f}")
    
    # 6. PageRank analysis
    print("\n5. PageRank Analysis...")
    pagerank = analyzer.calculate_pagerank()
    top_influencers = analyzer.top_influencers(method='pagerank', top_k=5)
    print("   Top 5 influencers (PageRank):")
    for node, score in top_influencers:
        print(f"     Node {node}: {score:.4f}")
    
    # 7. Link Prediction
    print("\n6. Link Prediction Analysis...")
    print("   Predicting potential connections...")
    
    # Sample some non-edges
    non_edges = list(nx.non_edges(G))[:10]
    print(f"   Analyzing {len(non_edges)} potential connections...")
    
    predictions = []
    for u, v in non_edges:
        cn = analyzer.common_neighbors_score(u, v)
        aa = analyzer.adamic_adar_score(u, v)
        predictions.append((u, v, cn, aa))
    
    # Sort by Adamic-Adar score
    predictions.sort(key=lambda x: x[3], reverse=True)
    print("\n   Top 5 predicted links (by Adamic-Adar):")
    for u, v, cn, aa in predictions[:5]:
        print(f"     ({u}, {v}): Common Neighbors={cn:.0f}, Adamic-Adar={aa:.4f}")
    
    # 8. Structural analysis
    print("\n7. Structural Analysis...")
    bridges = analyzer.identify_bridges()
    print(f"   - Number of bridges: {len(bridges)}")
    if bridges:
        print(f"   - Bridge edges: {bridges[:5]}...")  # Show first 5
    
    # 9. Visualizations
    print("\n8. Creating Visualizations...")
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Network with communities
    print("   - Network with communities (Matplotlib)...")
    visualize_network_matplotlib(
        G,
        communities=communities_louvain,
        centrality=pagerank,
        save_path='results/social_network_communities.png',
        title='Social Network - Community Structure'
    )
    
    # Community visualization
    print("   - Community structure analysis...")
    visualize_communities(
        G,
        communities_louvain,
        save_path='results/social_network_community_analysis.html',
        title='Community Structure Analysis'
    )
    
    # Centrality comparison
    print("   - Centrality comparison...")
    visualize_centrality_comparison(
        G,
        {
            'degree': centrality_results['degree'],
            'betweenness': centrality_results['betweenness'],
            'pagerank': centrality_results['pagerank']
        },
        save_path='results/social_network_centrality_comparison.png',
        title='Centrality Measures Comparison'
    )
    
    # 10. Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Communities: {len(communities_louvain)}")
    print(f"Top influencer: Node {top_influencers[0][0]} (PageRank: {top_influencers[0][1]:.4f})")
    print(f"Bridges: {len(bridges)}")
    print(f"\nVisualization saved to 'results/' directory")
    print("="*60 + "\n")
    
    print("✓ Social network analysis complete!")


if __name__ == '__main__':
    main()
