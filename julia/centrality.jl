# Graph Centrality Analysis Module
# Author: Gabriel Demetrios Lafis

using Graphs
using LinearAlgebra

"""
Calculate betweenness centrality for all nodes in a graph.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Float64}`: Betweenness centrality scores
"""
function betweenness_centrality(g::AbstractGraph)
    n = nv(g)
    centrality = zeros(Float64, n)
    
    for s in vertices(g)
        # BFS from source s
        stack = Int[]
        paths = Dict{Int, Vector{Int}}()
        sigma = zeros(Int, n)
        dist = fill(-1, n)
        
        sigma[s] = 1
        dist[s] = 0
        queue = [s]
        
        while !isempty(queue)
            v = popfirst!(queue)
            push!(stack, v)
            
            for w in neighbors(g, v)
                # First time we see w?
                if dist[w] < 0
                    push!(queue, w)
                    dist[w] = dist[v] + 1
                end
                
                # Shortest path to w via v?
                if dist[w] == dist[v] + 1
                    sigma[w] += sigma[v]
                    if !haskey(paths, w)
                        paths[w] = Int[]
                    end
                    push!(paths[w], v)
                end
            end
        end
        
        # Accumulation
        delta = zeros(Float64, n)
        while !isempty(stack)
            w = pop!(stack)
            if haskey(paths, w)
                for v in paths[w]
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                end
            end
            if w != s
                centrality[w] += delta[w]
            end
        end
    end
    
    # Normalize
    if n > 2
        centrality ./= ((n - 1) * (n - 2))
    end
    
    return Dict(i => centrality[i] for i in 1:n)
end

"""
Calculate closeness centrality for all nodes.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Float64}`: Closeness centrality scores
"""
function closeness_centrality(g::AbstractGraph)
    n = nv(g)
    centrality = Dict{Int, Float64}()
    
    for v in vertices(g)
        # BFS to find shortest paths
        dist = fill(typemax(Int), n)
        dist[v] = 0
        queue = [v]
        
        while !isempty(queue)
            u = popfirst!(queue)
            for w in neighbors(g, u)
                if dist[w] == typemax(Int)
                    dist[w] = dist[u] + 1
                    push!(queue, w)
                end
            end
        end
        
        # Calculate closeness
        reachable = filter(d -> d < typemax(Int), dist)
        if length(reachable) > 1
            centrality[v] = (length(reachable) - 1) / sum(reachable)
        else
            centrality[v] = 0.0
        end
    end
    
    return centrality
end

"""
Calculate eigenvector centrality using power iteration.

# Arguments
- `g::AbstractGraph`: Input graph
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Returns
- `Dict{Int, Float64}`: Eigenvector centrality scores
"""
function eigenvector_centrality(g::AbstractGraph; max_iter=100, tol=1e-6)
    n = nv(g)
    
    # Initialize with uniform distribution
    x = ones(Float64, n) / n
    
    # Power iteration
    for _ in 1:max_iter
        x_new = zeros(Float64, n)
        
        for v in vertices(g)
            for u in neighbors(g, v)
                x_new[v] += x[u]
            end
        end
        
        # Normalize
        norm_val = norm(x_new)
        if norm_val > 0
            x_new ./= norm_val
        end
        
        # Check convergence
        if norm(x_new - x) < tol
            x = x_new
            break
        end
        
        x = x_new
    end
    
    return Dict(i => x[i] for i in 1:n)
end

"""
Calculate degree centrality.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Float64}`: Degree centrality scores
"""
function degree_centrality(g::AbstractGraph)
    n = nv(g)
    centrality = Dict{Int, Float64}()
    
    for v in vertices(g)
        centrality[v] = degree(g, v) / (n - 1)
    end
    
    return centrality
end

