# Link Prediction Module
# Author: Gabriel Demetrios Lafis

using Graphs

"""
Common neighbors score for link prediction.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Float64`: Number of common neighbors
"""
function common_neighbors(g::AbstractGraph, u::Int, v::Int)
    neighbors_u = Set(neighbors(g, u))
    neighbors_v = Set(neighbors(g, v))
    return Float64(length(intersect(neighbors_u, neighbors_v)))
end

"""
Jaccard coefficient for link prediction.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Float64`: Jaccard coefficient score
"""
function jaccard_coefficient(g::AbstractGraph, u::Int, v::Int)
    neighbors_u = Set(neighbors(g, u))
    neighbors_v = Set(neighbors(g, v))
    
    union_size = length(union(neighbors_u, neighbors_v))
    if union_size == 0
        return 0.0
    end
    
    intersection_size = length(intersect(neighbors_u, neighbors_v))
    return intersection_size / union_size
end

"""
Adamic-Adar index for link prediction.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Float64`: Adamic-Adar score
"""
function adamic_adar(g::AbstractGraph, u::Int, v::Int)
    neighbors_u = Set(neighbors(g, u))
    neighbors_v = Set(neighbors(g, v))
    common = intersect(neighbors_u, neighbors_v)
    
    score = 0.0
    for w in common
        deg = degree(g, w)
        if deg > 1
            score += 1.0 / log(deg)
        end
    end
    
    return score
end

"""
Preferential attachment score for link prediction.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Float64`: Product of node degrees
"""
function preferential_attachment(g::AbstractGraph, u::Int, v::Int)
    return Float64(degree(g, u) * degree(g, v))
end

"""
Resource allocation index for link prediction.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Float64`: Resource allocation score
"""
function resource_allocation(g::AbstractGraph, u::Int, v::Int)
    neighbors_u = Set(neighbors(g, u))
    neighbors_v = Set(neighbors(g, v))
    common = intersect(neighbors_u, neighbors_v)
    
    score = 0.0
    for w in common
        deg = degree(g, w)
        if deg > 0
            score += 1.0 / deg
        end
    end
    
    return score
end

"""
Calculate all link prediction scores for a pair of nodes.

# Arguments
- `g::AbstractGraph`: Input graph
- `u::Int`: First node
- `v::Int`: Second node

# Returns
- `Dict{String, Float64}`: Dictionary of all scores
"""
function all_link_scores(g::AbstractGraph, u::Int, v::Int)
    return Dict(
        "common_neighbors" => common_neighbors(g, u, v),
        "jaccard" => jaccard_coefficient(g, u, v),
        "adamic_adar" => adamic_adar(g, u, v),
        "preferential_attachment" => preferential_attachment(g, u, v),
        "resource_allocation" => resource_allocation(g, u, v)
    )
end

"""
Predict top k most likely links in the graph.

# Arguments
- `g::AbstractGraph`: Input graph
- `k::Int`: Number of links to predict
- `method::String`: Link prediction method (default: "adamic_adar")

# Returns
- `Vector{Tuple{Int, Int, Float64}}`: Top k predicted links with scores
"""
function predict_top_links(g::AbstractGraph, k::Int=10; method="adamic_adar")
    n = nv(g)
    scores = Vector{Tuple{Int, Int, Float64}}()
    
    # Choose scoring function
    score_func = if method == "common_neighbors"
        common_neighbors
    elseif method == "jaccard"
        jaccard_coefficient
    elseif method == "adamic_adar"
        adamic_adar
    elseif method == "preferential_attachment"
        preferential_attachment
    elseif method == "resource_allocation"
        resource_allocation
    else
        error("Unknown method: $method")
    end
    
    # Calculate scores for all non-edges
    for u in vertices(g)
        for v in (u+1):n
            if !has_edge(g, u, v)
                score = score_func(g, u, v)
                push!(scores, (u, v, score))
            end
        end
    end
    
    # Sort by score and return top k
    sort!(scores, by=x -> x[3], rev=true)
    return scores[1:min(k, length(scores))]
end

