# Community Detection Module
# Author: Gabriel Demetrios Lafis

using Graphs
using DataStructures

"""
Label Propagation Algorithm for community detection.

# Arguments
- `g::AbstractGraph`: Input graph
- `max_iter::Int`: Maximum iterations (default: 100)

# Returns
- `Dict{Int, Int}`: Community assignment for each node
"""
function label_propagation(g::AbstractGraph; max_iter=100)
    n = nv(g)
    
    # Initialize each node with unique label
    labels = Dict(i => i for i in 1:n)
    
    # Iterate until convergence or max iterations
    for iter in 1:max_iter
        changed = false
        
        # Process nodes in random order
        nodes = shuffle(collect(vertices(g)))
        
        for v in nodes
            # Count labels of neighbors
            neighbor_labels = counter(Int)
            
            for u in neighbors(g, v)
                inc!(neighbor_labels, labels[u])
            end
            
            # Assign most common label
            if !isempty(neighbor_labels)
                most_common_label = argmax(neighbor_labels)
                
                if labels[v] != most_common_label
                    labels[v] = most_common_label
                    changed = true
                end
            end
        end
        
        # Stop if no changes
        if !changed
            break
        end
    end
    
    return labels
end

"""
Calculate modularity of a graph partition.

# Arguments
- `g::AbstractGraph`: Input graph
- `communities::Dict{Int, Int}`: Community assignments

# Returns
- `Float64`: Modularity score
"""
function modularity(g::AbstractGraph, communities::Dict{Int, Int})
    m = ne(g)
    if m == 0
        return 0.0
    end
    
    Q = 0.0
    degrees = degree(g)
    
    for i in vertices(g)
        for j in vertices(g)
            if communities[i] == communities[j]
                A_ij = has_edge(g, i, j) ? 1.0 : 0.0
                expected = (degrees[i] * degrees[j]) / (2 * m)
                Q += A_ij - expected
            end
        end
    end
    
    return Q / (2 * m)
end

"""
Greedy modularity optimization for community detection.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Int}`: Community assignment for each node
"""
function greedy_modularity(g::AbstractGraph)
    n = nv(g)
    
    # Initialize each node in its own community
    communities = Dict(i => i for i in 1:n)
    
    # Calculate initial modularity
    best_modularity = modularity(g, communities)
    improved = true
    
    while improved
        improved = false
        
        for v in vertices(g)
            current_community = communities[v]
            best_community = current_community
            
            # Try moving node to each neighbor's community
            neighbor_communities = Set(communities[u] for u in neighbors(g, v))
            
            for comm in neighbor_communities
                # Temporarily move node
                communities[v] = comm
                new_modularity = modularity(g, communities)
                
                if new_modularity > best_modularity
                    best_modularity = new_modularity
                    best_community = comm
                    improved = true
                end
            end
            
            # Keep best assignment
            communities[v] = best_community
        end
    end
    
    return communities
end

"""
Find connected components (basic community structure).

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Vector{Set{Int}}`: List of connected components
"""
function find_connected_components(g::AbstractGraph)
    n = nv(g)
    visited = falses(n)
    components = Vector{Set{Int}}()
    
    for start_node in vertices(g)
        if !visited[start_node]
            # BFS to find component
            component = Set{Int}()
            queue = [start_node]
            visited[start_node] = true
            
            while !isempty(queue)
                v = popfirst!(queue)
                push!(component, v)
                
                for u in neighbors(g, v)
                    if !visited[u]
                        visited[u] = true
                        push!(queue, u)
                    end
                end
            end
            
            push!(components, component)
        end
    end
    
    return components
end

