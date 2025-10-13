# Transportation Network Analysis Example
# Author: Gabriel Demetrios Lafis

using Graphs
using Printf
using Statistics

# Include modules
include("../julia/centrality.jl")
include("../julia/shortest_paths.jl")
include("../julia/utils.jl")

"""
Create sample city road network
"""
function create_road_network()
    # Create network representing city streets
    g = SimpleGraph(30)
    
    # Main highways (grid structure)
    for i in 1:5
        for j in 1:5
            node = (i-1)*5 + j
            # Horizontal connections
            if j < 5
                add_edge!(g, node, node + 1)
            end
            # Vertical connections
            if i < 5
                add_edge!(g, node, node + 5)
            end
        end
    end
    
    # Add diagonal shortcuts
    add_edge!(g, 1, 7)
    add_edge!(g, 3, 9)
    add_edge!(g, 11, 17)
    add_edge!(g, 13, 19)
    add_edge!(g, 17, 23)
    add_edge!(g, 19, 25)
    
    # Connect to peripheral areas
    for i in 26:30
        add_edge!(g, 25, i)
    end
    
    return g
end

"""
Analyze transportation network
"""
function main()
    println("\n" * "="^60)
    println("TRANSPORTATION NETWORK ANALYSIS")
    println("="^60 * "\n")
    
    # 1. Create road network
    println("1. Creating City Road Network...")
    g = create_road_network()
    println("   Network: $(nv(g)) intersections, $(ne(g)) roads")
    
    # 2. Network metrics
    println("\n2. Network Topology Analysis...")
    density = graph_density(g)
    avg_deg = mean(degree(g))
    @printf("   Density: %.4f\n", density)
    @printf("   Average degree: %.2f\n", avg_deg)
    
    # Connectivity
    is_conn = is_connected(g)
    println("   Connected: $is_conn")
    
    if is_conn
        diam = graph_diameter(g)
        rad = graph_radius(g)
        println("   Diameter: $diam")
        println("   Radius: $rad")
    end
    
    # 3. Identify critical intersections
    println("\n3. Identifying Critical Intersections...")
    
    # Betweenness centrality - traffic flow through intersection
    bc = betweenness_centrality(g)
    sorted_bc = sort(collect(bc), by=x->x[2], rev=true)
    
    println("   Top 10 critical intersections (betweenness):")
    for i in 1:min(10, length(sorted_bc))
        node, score = sorted_bc[i]
        @printf("     Intersection %2d: %.4f\n", node, score)
    end
    
    # Degree centrality - number of connected roads
    dc = degree_centrality(g)
    sorted_dc = sort(collect(dc), by=x->x[2], rev=true)
    
    println("\n   Top 10 busiest intersections (degree):")
    for i in 1:min(10, length(sorted_dc))
        node, score = sorted_dc[i]
        deg = degree(g, node)
        @printf("     Intersection %2d: %d roads (centrality: %.4f)\n", node, deg, score)
    end
    
    # 4. Shortest paths analysis
    println("\n4. Route Planning (Shortest Paths)...")
    
    # Example routes
    routes = [
        (1, 25),   # Northwest to Southeast
        (5, 21),   # Northeast to Southwest
        (13, 30),  # Center to periphery
    ]
    
    println("   Sample routes:")
    for (start, finish) in routes
        dist, pred = dijkstra(g, start)
        
        # Reconstruct path
        path = Int[]
        current = finish
        while current != 0
            pushfirst!(path, current)
            current = pred[current]
        end
        
        @printf("     Route %2d → %2d: Distance = %.0f, Path = %s\n",
                start, finish, dist[finish], path)
    end
    
    # 5. Identify critical roads (bridges)
    println("\n5. Critical Roads Analysis...")
    bridges = find_bridges(g)
    println("   Number of critical roads (bridges): $(length(bridges))")
    if !isempty(bridges)
        println("   Critical roads:")
        for (u, v) in bridges[1:min(5, length(bridges))]
            println("     Road ($u, $v)")
        end
    end
    
    # 6. Articulation points (critical intersections)
    println("\n6. Critical Intersections (Articulation Points)...")
    art_points = find_articulation_points(g)
    println("   Number of critical intersections: $(length(art_points))")
    if !isempty(art_points)
        println("   Critical intersections: $(art_points[1:min(5, length(art_points))])")
    end
    
    # 7. Traffic flow simulation
    println("\n7. Traffic Flow Simulation...")
    println("   Simulating removal of busiest intersection...")
    
    busiest = sorted_dc[1][1]
    println("   Removing intersection $busiest...")
    
    # Create network without busiest intersection
    g_modified = SimpleGraph(nv(g) - 1)
    vertex_map = Dict{Int, Int}()
    new_id = 1
    for v in vertices(g)
        if v != busiest
            vertex_map[v] = new_id
            new_id += 1
        end
    end
    
    for edge in edges(g)
        u, w = src(edge), dst(edge)
        if u != busiest && w != busiest
            add_edge!(g_modified, vertex_map[u], vertex_map[w])
        end
    end
    
    is_conn_after = is_connected(g_modified)
    println("   Network still connected: $is_conn_after")
    
    if is_conn_after
        # Calculate average shortest path before and after
        diam_after = graph_diameter(g_modified)
        println("   Diameter after removal: $diam_after")
    else
        num_components = length(connected_components(g_modified))
        println("   Network fragmented into $num_components components")
    end
    
    # 8. Alternative routes analysis
    println("\n8. Alternative Routes Analysis...")
    source, target = 1, 25
    
    # Find shortest path
    dist, pred = dijkstra(g, source)
    
    # Reconstruct path
    shortest_path = Int[]
    current = target
    while current != 0
        pushfirst!(shortest_path, current)
        current = pred[current]
    end
    
    println("   Primary route ($(source) → $(target)): $shortest_path")
    println("   Distance: $(dist[target])")
    
    # Remove an edge from shortest path to find alternative
    if length(shortest_path) >= 2
        u, v = shortest_path[1], shortest_path[2]
        g_alt = copy(g)
        rem_edge!(g_alt, u, v)
        
        dist_alt, pred_alt = dijkstra(g_alt, source)
        
        # Reconstruct alternative path
        alt_path = Int[]
        current = target
        while current != 0
            pushfirst!(alt_path, current)
            current = pred_alt[current]
        end
        
        println("   Alternative route (avoiding road $u-$v): $alt_path")
        println("   Distance: $(dist_alt[target])")
        @printf("   Increase: %.1f%%\n", 100 * (dist_alt[target] - dist[target]) / dist[target])
    end
    
    # 9. Summary
    println("\n" * "="^60)
    println("ANALYSIS SUMMARY")
    println("="^60)
    println("Intersections: $(nv(g))")
    println("Roads: $(ne(g))")
    println("Critical intersections: $(length(art_points))")
    println("Critical roads: $(length(bridges))")
    println("Most critical intersection: $(sorted_bc[1][1])")
    println("Busiest intersection: $(sorted_dc[1][1])")
    println("="^60 * "\n")
    
    println("✓ Transportation network analysis complete!")
end

# Run analysis
main()
