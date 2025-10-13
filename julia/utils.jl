# Utility Functions Module
# Author: Gabriel Demetrios Lafis

using Graphs
using LinearAlgebra
using SparseArrays
using Statistics

"""
Convert edge list to graph.

# Arguments
- `edges::Vector{Tuple{Int, Int}}`: List of edges
- `n::Int`: Number of nodes (optional)

# Returns
- `SimpleGraph`: Created graph
"""
function edgelist_to_graph(edges::Vector{Tuple{Int, Int}}, n::Int=0)
    if n == 0
        # Determine number of nodes from edges
        n = maximum(max(e[1], e[2]) for e in edges)
    end
    
    g = SimpleGraph(n)
    for (u, v) in edges
        add_edge!(g, u, v)
    end
    
    return g
end

"""
Convert adjacency matrix to graph.

# Arguments
- `adj_matrix::Matrix`: Adjacency matrix

# Returns
- `SimpleGraph`: Created graph
"""
function adjacency_matrix_to_graph(adj_matrix::Matrix)
    n = size(adj_matrix, 1)
    g = SimpleGraph(n)
    
    for i in 1:n
        for j in (i+1):n
            if adj_matrix[i, j] != 0
                add_edge!(g, i, j)
            end
        end
    end
    
    return g
end

"""
Calculate graph density.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Float64`: Graph density
"""
function graph_density(g::AbstractGraph)
    n = nv(g)
    m = ne(g)
    
    if n <= 1
        return 0.0
    end
    
    max_edges = n * (n - 1) / 2
    return m / max_edges
end

"""
Find all triangles in the graph.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Vector{Tuple{Int, Int, Int}}`: List of triangles
"""
function find_triangles(g::AbstractGraph)
    triangles = Vector{Tuple{Int, Int, Int}}()
    
    for u in vertices(g)
        neighbors_u = neighbors(g, u)
        for i in 1:length(neighbors_u)
            v = neighbors_u[i]
            if v > u
                for j in (i+1):length(neighbors_u)
                    w = neighbors_u[j]
                    if w > v && has_edge(g, v, w)
                        push!(triangles, (u, v, w))
                    end
                end
            end
        end
    end
    
    return triangles
end

"""
Count triangles in the graph.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Int`: Number of triangles
"""
function count_triangles(g::AbstractGraph)
    count = 0
    
    for u in vertices(g)
        neighbors_u = Set(neighbors(g, u))
        for v in neighbors_u
            if v > u
                for w in neighbors_u
                    if w > v && has_edge(g, v, w)
                        count += 1
                    end
                end
            end
        end
    end
    
    return count
end

"""
Calculate local clustering coefficient for a node.

# Arguments
- `g::AbstractGraph`: Input graph
- `v::Int`: Node

# Returns
- `Float64`: Local clustering coefficient
"""
function local_clustering(g::AbstractGraph, v::Int)
    neighbors_v = neighbors(g, v)
    k = length(neighbors_v)
    
    if k < 2
        return 0.0
    end
    
    # Count edges between neighbors
    edges_between = 0
    for i in 1:k
        for j in (i+1):k
            if has_edge(g, neighbors_v[i], neighbors_v[j])
                edges_between += 1
            end
        end
    end
    
    return 2 * edges_between / (k * (k - 1))
end

"""
Calculate average clustering coefficient.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Float64`: Average clustering coefficient
"""
function average_clustering(g::AbstractGraph)
    n = nv(g)
    if n == 0
        return 0.0
    end
    
    total = 0.0
    for v in vertices(g)
        total += local_clustering(g, v)
    end
    
    return total / n
end

"""
Find bridges (edges whose removal disconnects the graph).

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Vector{Tuple{Int, Int}}`: List of bridges
"""
function find_bridges(g::AbstractGraph)
    bridges = Vector{Tuple{Int, Int}}()
    original_components = length(connected_components(g))
    
    for edge in edges(g)
        u, v = src(edge), dst(edge)
        
        # Create temporary graph without this edge
        g_temp = copy(g)
        rem_edge!(g_temp, u, v)
        
        # Check if removing edge increases components
        if length(connected_components(g_temp)) > original_components
            push!(bridges, (u, v))
        end
    end
    
    return bridges
end

"""
Find articulation points (nodes whose removal disconnects the graph).

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Vector{Int}`: List of articulation points
"""
function find_articulation_points(g::AbstractGraph)
    n = nv(g)
    articulation_points = Vector{Int}()
    original_components = length(connected_components(g))
    
    for v in vertices(g)
        # Create temporary graph without this vertex
        g_temp = SimpleGraph(n - 1)
        
        # Map old vertices to new (skipping v)
        vertex_map = Dict{Int, Int}()
        new_id = 1
        for u in vertices(g)
            if u != v
                vertex_map[u] = new_id
                new_id += 1
            end
        end
        
        # Add edges (excluding those involving v)
        for edge in edges(g)
            u, w = src(edge), dst(edge)
            if u != v && w != v
                add_edge!(g_temp, vertex_map[u], vertex_map[w])
            end
        end
        
        # Check if removing vertex increases components
        if length(connected_components(g_temp)) > original_components
            push!(articulation_points, v)
        end
    end
    
    return articulation_points
end

"""
Calculate graph diameter (longest shortest path).

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Float64`: Graph diameter (Inf if disconnected)
"""
function graph_diameter(g::AbstractGraph)
    if !is_connected(g)
        return Inf
    end
    
    n = nv(g)
    max_dist = 0.0
    
    for u in vertices(g)
        dist = fill(Inf, n)
        dist[u] = 0
        queue = [u]
        
        while !isempty(queue)
            v = popfirst!(queue)
            for w in neighbors(g, v)
                if dist[w] == Inf
                    dist[w] = dist[v] + 1
                    push!(queue, w)
                    max_dist = max(max_dist, dist[w])
                end
            end
        end
    end
    
    return max_dist
end

"""
Calculate graph radius (minimum eccentricity).

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Float64`: Graph radius
"""
function graph_radius(g::AbstractGraph)
    if !is_connected(g)
        return Inf
    end
    
    n = nv(g)
    min_eccentricity = Inf
    
    for u in vertices(g)
        # Find eccentricity (max distance from u to any other node)
        dist = fill(Inf, n)
        dist[u] = 0
        queue = [u]
        max_dist = 0.0
        
        while !isempty(queue)
            v = popfirst!(queue)
            for w in neighbors(g, v)
                if dist[w] == Inf
                    dist[w] = dist[v] + 1
                    push!(queue, w)
                    max_dist = max(max_dist, dist[w])
                end
            end
        end
        
        min_eccentricity = min(min_eccentricity, max_dist)
    end
    
    return min_eccentricity
end

"""
Generate random graph with given properties.

# Arguments
- `n::Int`: Number of nodes
- `p::Float64`: Edge probability (Erdős-Rényi)

# Returns
- `SimpleGraph`: Random graph
"""
function random_graph(n::Int, p::Float64)
    g = SimpleGraph(n)
    
    for i in 1:n
        for j in (i+1):n
            if rand() < p
                add_edge!(g, i, j)
            end
        end
    end
    
    return g
end

