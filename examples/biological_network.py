"""
Biological Network Analysis Example

Analyzes protein-protein interaction networks to identify:
- Essential proteins (high centrality)
- Functional modules (communities)
- Interaction patterns

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import networkx as nx
from graph_analyzer import GraphAnalyzer
from metrics import NetworkMetrics
import matplotlib.pyplot as plt

def create_sample_protein_network():
    """Create sample protein-protein interaction network."""
    G = nx.Graph()
    
    # Add proteins as nodes
    proteins = list(range(1, 51))
    G.add_nodes_from(proteins)
    
    # Add interactions (edges)
    # Module 1: DNA repair proteins
    module1 = list(range(1, 11))
    for i in range(len(module1)):
        for j in range(i+1, len(module1)):
            if abs(i - j) <= 2:  # Local interactions
                G.add_edge(module1[i], module1[j])
    
    # Module 2: Cell cycle proteins
    module2 = list(range(11, 21))
    for i in range(len(module2)):
        for j in range(i+1, len(module2)):
            if abs(i - j) <= 2:
                G.add_edge(module2[i], module2[j])
    
    # Module 3: Signal transduction
    module3 = list(range(21, 31))
    for i in range(len(module3)):
        for j in range(i+1, len(module3)):
            if abs(i - j) <= 2:
                G.add_edge(module3[i], module3[j])
    
    # Hub proteins connecting modules
    hub_proteins = [31, 32, 33]
    for hub in hub_proteins:
        # Connect to multiple modules
        G.add_edge(hub, 5)   # Module 1
        G.add_edge(hub, 15)  # Module 2
        G.add_edge(hub, 25)  # Module 3
    
    # Add some random interactions
    import random
    random.seed(42)
    for _ in range(20):
        u, v = random.sample(proteins, 2)
        G.add_edge(u, v)
    
    return G


def main():
    print("\n" + "="*60)
    print("PROTEIN-PROTEIN INTERACTION NETWORK ANALYSIS")
    print("="*60 + "\n")
    
    # 1. Create/load network
    print("1. Creating Protein Network...")
    G = create_sample_protein_network()
    print(f"   Network: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")
    
    analyzer = GraphAnalyzer(G)
    
    # 2. Network topology analysis
    print("\n2. Network Topology Analysis...")
    metrics = NetworkMetrics.all_metrics(G)
    
    print("   Basic properties:")
    for key, value in metrics['basic'].items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")
    
    print("\n   Clustering:")
    for key, value in metrics['clustering'].items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")
    
    # 3. Identify essential proteins (high centrality)
    print("\n3. Identifying Essential Proteins...")
    centralities = analyzer.calculate_all_centralities()
    
    # Combine different centrality measures
    combined_score = {}
    for node in G.nodes():
        score = (
            centralities['degree'].get(node, 0) +
            centralities['betweenness'].get(node, 0) +
            centralities['closeness'].get(node, 0)
        ) / 3
        combined_score[node] = score
    
    essential_proteins = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)[:10]
    print("   Top 10 essential proteins (combined centrality):")
    for protein, score in essential_proteins:
        print(f"     Protein {protein}: {score:.4f}")
    
    # 4. Detect functional modules
    print("\n4. Detecting Functional Modules...")
    modules = analyzer.detect_communities(method='louvain')
    print(f"   {len(modules)} functional modules detected:")
    for i, module in enumerate(modules):
        print(f"     Module {i+1}: {len(module)} proteins")
        if len(module) <= 10:
            print(f"       Members: {sorted(module)}")
    
    # 5. Hub proteins analysis
    print("\n5. Hub Proteins Analysis...")
    degree_cent = centralities['degree']
    hubs = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("   Top 10 hub proteins (degree centrality):")
    for protein, cent in hubs:
        degree = G.degree(protein)
        print(f"     Protein {protein}: {degree} interactions (centrality: {cent:.4f})")
    
    # 6. Bottleneck proteins (high betweenness)
    print("\n6. Bottleneck Proteins Analysis...")
    betweenness = centralities['betweenness']
    bottlenecks = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("   Top 10 bottleneck proteins (betweenness centrality):")
    for protein, cent in bottlenecks:
        print(f"     Protein {protein}: {cent:.4f}")
    
    # 7. Network resilience
    print("\n7. Network Resilience Analysis...")
    bridges = analyzer.identify_bridges()
    print(f"   Critical interactions (bridges): {len(bridges)}")
    
    # Simulate protein knockout
    print("\n   Simulating knockout of top hub protein...")
    top_hub = hubs[0][0]
    G_knockout = G.copy()
    G_knockout.remove_node(top_hub)
    
    components_before = nx.number_connected_components(G)
    components_after = nx.number_connected_components(G_knockout)
    
    print(f"     Before knockout: {components_before} component(s)")
    print(f"     After knockout: {components_after} component(s)")
    print(f"     Largest component size: {len(max(nx.connected_components(G_knockout), key=len))} proteins")
    
    # 8. Degree distribution analysis
    print("\n8. Degree Distribution Analysis...")
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"   Average degree: {sum(degrees)/len(degrees):.2f}")
    print(f"   Max degree: {max(degrees)}")
    print(f"   Min degree: {min(degrees)}")
    
    # Check if scale-free (power law)
    high_degree = sum(1 for d in degrees if d > 10)
    print(f"   High-degree proteins (>10 interactions): {high_degree}")
    
    # 9. Create visualization
    print("\n9. Creating Visualizations...")
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Color by module
    node_colors = []
    node_to_module = {}
    for i, module in enumerate(modules):
        for node in module:
            node_to_module[node] = i
    
    for node in G.nodes():
        node_colors.append(node_to_module.get(node, 0))
    
    # Size by degree
    node_sizes = [50 + 20 * G.degree(n) for n in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          cmap=plt.cm.Set3, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6)
    
    plt.title('Protein-Protein Interaction Network\nColors: Functional Modules, Size: Degree',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/protein_network.png', dpi=300, bbox_inches='tight')
    print("   Network visualization saved to 'results/protein_network.png'")
    
    # 10. Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Proteins: {G.number_of_nodes()}")
    print(f"Interactions: {G.number_of_edges()}")
    print(f"Functional modules: {len(modules)}")
    print(f"Top essential protein: {essential_proteins[0][0]}")
    print(f"Top hub protein: {hubs[0][0]} ({G.degree(hubs[0][0])} interactions)")
    print(f"Critical interactions: {len(bridges)}")
    print("="*60 + "\n")
    
    print("✓ Biological network analysis complete!")


if __name__ == '__main__':
    main()
