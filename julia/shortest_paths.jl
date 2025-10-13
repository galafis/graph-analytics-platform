# Shortest Path Algorithms Module
# Author: Gabriel Demetrios Lafis

using Graphs
using DataStructures

"""
Dijkstra's shortest path algorithm.

# Arguments
- `g::AbstractGraph`: Input graph
- `source::Int`: Source node
- `weights::Dict{Tuple{Int,Int}, Float64}`: Optional edge weights

# Returns
- `Tuple{Dict{Int, Float64}, Dict{Int, Int}}`: (distances, predecessors)
"""
function dijkstra(g::AbstractGraph, source::Int; weights=nothing)
    n = nv(g)
    dist = fill(Inf, n)
    pred = fill(0, n)
    dist[source] = 0.0
    
    # Priority queue: (distance, node)
    pq = PriorityQueue{Int, Float64}()
    for v in vertices(g)
        pq[v] = dist[v]
    end
    
    while !isempty(pq)
        u = dequeue!(pq)
        
        for v in neighbors(g, u)
            # Get edge weight
            weight = 1.0
            if weights !== nothing && haskey(weights, (u, v))
                weight = weights[(u, v)]
            end
            
            alt = dist[u] + weight
            if alt < dist[v]
                dist[v] = alt
                pred[v] = u
                pq[v] = alt
            end
        end
    end
    
    return (Dict(i => dist[i] for i in 1:n), Dict(i => pred[i] for i in 1:n))
end

"""
Bellman-Ford algorithm for shortest paths (handles negative weights).

# Arguments
- `g::AbstractGraph`: Input graph
- `source::Int`: Source node
- `weights::Dict{Tuple{Int,Int}, Float64}`: Edge weights

# Returns
- `Tuple{Dict{Int, Float64}, Dict{Int, Int}, Bool}`: (distances, predecessors, no_negative_cycle)
"""
function bellman_ford(g::AbstractGraph, source::Int; weights=nothing)
    n = nv(g)
    dist = fill(Inf, n)
    pred = fill(0, n)
    dist[source] = 0.0
    
    # Relax edges n-1 times
    for _ in 1:(n-1)
        for u in vertices(g)
            for v in neighbors(g, u)
                weight = 1.0
                if weights !== nothing && haskey(weights, (u, v))
                    weight = weights[(u, v)]
                end
                
                if dist[u] + weight < dist[v]
                    dist[v] = dist[u] + weight
                    pred[v] = u
                end
            end
        end
    end
    
    # Check for negative cycles
    has_negative_cycle = false
    for u in vertices(g)
        for v in neighbors(g, u)
            weight = 1.0
            if weights !== nothing && haskey(weights, (u, v))
                weight = weights[(u, v)]
            end
            
            if dist[u] + weight < dist[v]
                has_negative_cycle = true
                break
            end
        end
        if has_negative_cycle
            break
        end
    end
    
    return (Dict(i => dist[i] for i in 1:n), Dict(i => pred[i] for i in 1:n), !has_negative_cycle)
end

"""
Floyd-Warshall algorithm for all-pairs shortest paths.

# Arguments
- `g::AbstractGraph`: Input graph
- `weights::Dict{Tuple{Int,Int}, Float64}`: Optional edge weights

# Returns
- `Matrix{Float64}`: Distance matrix
"""
function floyd_warshall(g::AbstractGraph; weights=nothing)
    n = nv(g)
    dist = fill(Inf, (n, n))
    
    # Initialize distances
    for i in 1:n
        dist[i, i] = 0.0
    end
    
    for u in vertices(g)
        for v in neighbors(g, u)
            weight = 1.0
            if weights !== nothing && haskey(weights, (u, v))
                weight = weights[(u, v)]
            end
            dist[u, v] = weight
        end
    end
    
    # Floyd-Warshall algorithm
    for k in 1:n
        for i in 1:n
            for j in 1:n
                if dist[i, k] + dist[k, j] < dist[i, j]
                    dist[i, j] = dist[i, k] + dist[k, j]
                end
            end
        end
    end
    
    return dist
end

"""
Calculate all pairs shortest paths using efficient method.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Tuple{Int,Int}, Vector{Int}}`: Dictionary of shortest paths
"""
function all_pairs_shortest_paths(g::AbstractGraph)
    n = nv(g)
    paths = Dict{Tuple{Int,Int}, Vector{Int}}()
    
    for source in vertices(g)
        dist, pred = dijkstra(g, source)
        
        for target in vertices(g)
            if dist[target] != Inf
                # Reconstruct path
                path = Int[]
                current = target
                
                while current != 0
                    pushfirst!(path, current)
                    current = pred[current]
                end
                
                paths[(source, target)] = path
            end
        end
    end
    
    return paths
end

"""
A* algorithm for shortest path with heuristic.

# Arguments
- `g::AbstractGraph`: Input graph
- `source::Int`: Source node
- `target::Int`: Target node
- `heuristic::Function`: Heuristic function h(node, target)
- `weights::Dict{Tuple{Int,Int}, Float64}`: Optional edge weights

# Returns
- `Vector{Int}`: Shortest path from source to target
"""
function astar(g::AbstractGraph, source::Int, target::Int, heuristic::Function; weights=nothing)
    n = nv(g)
    
    # g_score: cost from start to node
    g_score = fill(Inf, n)
    g_score[source] = 0.0
    
    # f_score: g_score + heuristic
    f_score = fill(Inf, n)
    f_score[source] = heuristic(source, target)
    
    # Track path
    came_from = Dict{Int, Int}()
    
    # Open set with priority queue
    open_set = PriorityQueue{Int, Float64}()
    open_set[source] = f_score[source]
    
    while !isempty(open_set)
        current = dequeue!(open_set)
        
        if current == target
            # Reconstruct path
            path = [current]
            while haskey(came_from, current)
                current = came_from[current]
                pushfirst!(path, current)
            end
            return path
        end
        
        for neighbor in neighbors(g, current)
            weight = 1.0
            if weights !== nothing && haskey(weights, (current, neighbor))
                weight = weights[(current, neighbor)]
            end
            
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score[neighbor]
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
                
                if !haskey(open_set, neighbor)
                    open_set[neighbor] = f_score[neighbor]
                end
            end
        end
    end
    
    # No path found
    return Int[]
end

