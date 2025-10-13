# PageRank and Link Analysis Module
# Author: Gabriel Demetrios Lafis

using Graphs
using LinearAlgebra
using SparseArrays

"""
Calculate PageRank scores for all nodes in a graph.

# Arguments
- `g::AbstractGraph`: Input graph
- `alpha::Float64`: Damping factor (default: 0.85)
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Returns
- `Dict{Int, Float64}`: PageRank scores for each node
"""
function pagerank(g::AbstractGraph; alpha=0.85, max_iter=100, tol=1e-6)
    n = nv(g)
    
    # Initialize PageRank vector
    pr = ones(Float64, n) / n
    
    # Build adjacency matrix
    A = adjacency_matrix(g)
    
    # Calculate out-degrees
    out_degrees = vec(sum(A, dims=2))
    
    # Handle dangling nodes (nodes with no outgoing edges)
    dangling = out_degrees .== 0
    
    # Normalize adjacency matrix by out-degrees
    D_inv = spdiagm(0 => [d > 0 ? 1/d : 0 for d in out_degrees])
    M = A' * D_inv
    
    # Power iteration
    for iter in 1:max_iter
        pr_new = zeros(Float64, n)
        
        # PageRank update
        pr_new = alpha * M * pr
        
        # Add contribution from dangling nodes
        dangling_sum = sum(pr[dangling])
        pr_new .+= alpha * dangling_sum / n
        
        # Add teleportation
        pr_new .+= (1 - alpha) / n
        
        # Check convergence
        if norm(pr_new - pr, 1) < tol
            pr = pr_new
            break
        end
        
        pr = pr_new
    end
    
    return Dict(i => pr[i] for i in 1:n)
end

"""
Calculate HITS (Hyperlink-Induced Topic Search) authority and hub scores.

# Arguments
- `g::AbstractGraph`: Input graph
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Returns
- `Tuple{Dict, Dict}`: (authority_scores, hub_scores)
"""
function hits(g::AbstractGraph; max_iter=100, tol=1e-6)
    n = nv(g)
    
    # Initialize authority and hub vectors
    auth = ones(Float64, n) / sqrt(n)
    hub = ones(Float64, n) / sqrt(n)
    
    # Build adjacency matrix
    A = adjacency_matrix(g)
    
    # Power iteration
    for iter in 1:max_iter
        # Update authority scores
        auth_new = A' * hub
        auth_new ./= norm(auth_new)
        
        # Update hub scores
        hub_new = A * auth_new
        hub_new ./= norm(hub_new)
        
        # Check convergence
        if norm(auth_new - auth) < tol && norm(hub_new - hub) < tol
            auth = auth_new
            hub = hub_new
            break
        end
        
        auth = auth_new
        hub = hub_new
    end
    
    auth_dict = Dict(i => auth[i] for i in 1:n)
    hub_dict = Dict(i => hub[i] for i in 1:n)
    
    return (auth_dict, hub_dict)
end

"""
Calculate personalized PageRank from a set of source nodes.

# Arguments
- `g::AbstractGraph`: Input graph
- `sources::Vector{Int}`: Source nodes for personalization
- `alpha::Float64`: Damping factor (default: 0.85)
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Returns
- `Dict{Int, Float64}`: Personalized PageRank scores
"""
function personalized_pagerank(
    g::AbstractGraph,
    sources::Vector{Int};
    alpha=0.85,
    max_iter=100,
    tol=1e-6
)
    n = nv(g)
    
    # Initialize PageRank vector
    pr = ones(Float64, n) / n
    
    # Create personalization vector
    personalization = zeros(Float64, n)
    for s in sources
        personalization[s] = 1.0 / length(sources)
    end
    
    # Build adjacency matrix
    A = adjacency_matrix(g)
    
    # Calculate out-degrees
    out_degrees = vec(sum(A, dims=2))
    
    # Normalize adjacency matrix
    D_inv = spdiagm(0 => [d > 0 ? 1/d : 0 for d in out_degrees])
    M = A' * D_inv
    
    # Power iteration
    for iter in 1:max_iter
        pr_new = alpha * M * pr + (1 - alpha) * personalization
        
        # Check convergence
        if norm(pr_new - pr, 1) < tol
            pr = pr_new
            break
        end
        
        pr = pr_new
    end
    
    return Dict(i => pr[i] for i in 1:n)
end

"""
Identify influential spreaders using k-shell decomposition.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Int}`: K-shell index for each node
"""
function kshell_decomposition(g::AbstractGraph)
    n = nv(g)
    degrees = degree(g)
    kshell = zeros(Int, n)
    
    # Create mutable copy of degrees
    current_degrees = copy(degrees)
    remaining = Set(1:n)
    shell = 1
    
    while !isempty(remaining)
        # Find nodes with minimum degree
        min_degree = minimum(current_degrees[i] for i in remaining)
        
        # Remove nodes with minimum degree
        to_remove = Set{Int}()
        for v in remaining
            if current_degrees[v] <= min_degree
                push!(to_remove, v)
                kshell[v] = shell
            end
        end
        
        # Update degrees of neighbors
        for v in to_remove
            for u in neighbors(g, v)
                if u in remaining && !(u in to_remove)
                    current_degrees[u] -= 1
                end
            end
        end
        
        # Remove nodes
        setdiff!(remaining, to_remove)
        
        # If all nodes at this level are removed, increment shell
        if isempty(remaining) || minimum(current_degrees[i] for i in remaining) > min_degree
            shell += 1
        end
    end
    
    return Dict(i => kshell[i] for i in 1:n)
end

