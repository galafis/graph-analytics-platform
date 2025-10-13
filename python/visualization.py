"""
Advanced Visualization Module

Provides comprehensive network visualization capabilities
including static plots, interactive visualizations, and animations.

Author: Gabriel Demetrios Lafis
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import warnings


def visualize_network_matplotlib(
    G: nx.Graph,
    communities: Optional[List[Set[int]]] = None,
    centrality: Optional[Dict[int, float]] = None,
    layout: str = 'spring',
    node_size: int = 300,
    save_path: Optional[str] = None,
    title: str = "Network Visualization",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create static network visualization using Matplotlib.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to visualize
    communities : list of sets, optional
        Community assignments
    centrality : dict, optional
        Centrality scores for node sizing
    layout : str
        Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
    node_size : int
        Base node size
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Determine node colors
    if communities:
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        node_colors = [node_to_comm.get(node, 0) for node in G.nodes()]
        cmap = plt.cm.Set3
    else:
        node_colors = [G.degree(node) for node in G.nodes()]
        cmap = plt.cm.viridis
    
    # Determine node sizes
    if centrality:
        node_sizes = [node_size * (1 + 5 * centrality.get(node, 0)) for node in G.nodes()]
    else:
        node_sizes = [node_size * (1 + G.degree(node) / 10) for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=cmap,
        alpha=0.8
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_network_plotly(
    G: nx.Graph,
    communities: Optional[List[Set[int]]] = None,
    centrality: Optional[Dict[int, float]] = None,
    save_path: Optional[str] = None,
    title: str = "Network Visualization"
) -> go.Figure:
    """
    Create interactive network visualization using Plotly.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to visualize
    communities : list of sets, optional
        Community assignments
    centrality : dict, optional
        Centrality scores for node sizing
    save_path : str, optional
        Path to save HTML file
    title : str
        Plot title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
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
        text = f'Node {node}<br>Degree: {degree}'
        if centrality:
            text += f'<br>Centrality: {centrality.get(node, 0):.4f}'
        node_text.append(text)
        
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
    
    return fig


def visualize_degree_distribution(
    G: nx.Graph,
    save_path: Optional[str] = None,
    title: str = "Degree Distribution"
):
    """
    Visualize degree distribution of the network.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    degrees = [G.degree(n) for n in G.nodes()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Degree Histogram', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Log-log plot
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_sorted]
    
    axes[1].loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
    axes[1].set_xlabel('Degree (log scale)', fontsize=12)
    axes[1].set_ylabel('Count (log scale)', fontsize=12)
    axes[1].set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_communities(
    G: nx.Graph,
    communities: List[Set[int]],
    save_path: Optional[str] = None,
    title: str = "Community Structure"
):
    """
    Visualize community structure with statistics.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : list of sets
        Detected communities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Community Sizes', 'Network with Communities'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Community sizes
    sizes = [len(comm) for comm in communities]
    fig.add_trace(
        go.Bar(x=list(range(len(communities))), y=sizes, name='Size'),
        row=1, col=1
    )
    
    # Network visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Node colors by community
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
    
    node_colors = [node_to_comm.get(node, 0) for node in G.nodes()]
    
    # Edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=10,
                color=node_colors,
                colorscale='Viridis',
                showscale=True
            ),
            hovertext=[f'Node {n}<br>Community {node_to_comm.get(n, 0)}' for n in G.nodes()],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(title_text=title, height=500)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def visualize_centrality_comparison(
    G: nx.Graph,
    centrality_measures: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
    title: str = "Centrality Comparison"
):
    """
    Compare different centrality measures.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    centrality_measures : dict
        Dictionary of centrality measures
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    n_measures = len(centrality_measures)
    fig, axes = plt.subplots(1, n_measures, figsize=(5 * n_measures, 5))
    
    if n_measures == 1:
        axes = [axes]
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    for idx, (measure_name, centrality) in enumerate(centrality_measures.items()):
        node_colors = [centrality.get(node, 0) for node in G.nodes()]
        node_sizes = [300 * (1 + centrality.get(node, 0)) for node in G.nodes()]
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=axes[idx])
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.viridis,
            alpha=0.8,
            ax=axes[idx]
        )
        
        axes[idx].set_title(measure_name.capitalize(), fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

