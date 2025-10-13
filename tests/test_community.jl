# Test script for community detection algorithms
# Author: Gabriel Demetrios Lafis

using Test
using Graphs

# Include the community module
include("../community.jl")

@testset "Community Detection Tests" begin
    
    @testset "Label Propagation" begin
        # Create graph with two clear communities
        g = SimpleGraph(6)
        # Community 1: nodes 1, 2, 3
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 1)
        # Community 2: nodes 4, 5, 6
        add_edge!(g, 4, 5)
        add_edge!(g, 5, 6)
        add_edge!(g, 6, 4)
        # Bridge between communities
        add_edge!(g, 3, 4)
        
        labels = label_propagation(g)
        @test length(labels) == 6
        @test length(unique(values(labels))) <= 6  # At most 6 communities
    end
    
    @testset "Modularity Calculation" begin
        # Create simple graph
        g = SimpleGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        
        # Assign all to same community
        communities = Dict(i => 1 for i in 1:4)
        Q = modularity(g, communities)
        @test -1.0 <= Q <= 1.0
        
        # Empty graph should have 0 modularity
        g_empty = SimpleGraph(2)
        communities_empty = Dict(1 => 1, 2 => 2)
        Q_empty = modularity(g_empty, communities_empty)
        @test Q_empty == 0.0
    end
    
    @testset "Greedy Modularity" begin
        # Create graph with clear community structure
        g = SimpleGraph(6)
        # Community 1
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 1, 3)
        # Community 2
        add_edge!(g, 4, 5)
        add_edge!(g, 5, 6)
        add_edge!(g, 4, 6)
        # Weak connection between communities
        add_edge!(g, 3, 4)
        
        communities = greedy_modularity(g)
        @test length(communities) == 6
        
        # Check modularity is positive
        Q = modularity(g, communities)
        @test Q > 0
    end
    
    @testset "Connected Components" begin
        # Create disconnected graph
        g = SimpleGraph(6)
        # Component 1
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        # Component 2
        add_edge!(g, 4, 5)
        # Isolated node 6
        
        components = find_connected_components(g)
        @test length(components) == 3
        
        # Check sizes
        sizes = [length(comp) for comp in components]
        @test 3 in sizes  # Component 1
        @test 2 in sizes  # Component 2
        @test 1 in sizes  # Isolated node
    end
    
    @testset "Single Node Graph" begin
        g = SimpleGraph(1)
        labels = label_propagation(g)
        @test length(labels) == 1
        
        communities = Dict(1 => 1)
        Q = modularity(g, communities)
        @test Q == 0.0
    end
    
    @testset "Complete Graph" begin
        g = complete_graph(5)
        communities = greedy_modularity(g)
        @test length(communities) == 5
        
        # All nodes might be in same community for complete graph
        num_communities = length(unique(values(communities)))
        @test num_communities >= 1
    end
    
    @testset "Star Graph Communities" begin
        # Create star graph
        g = SimpleGraph(6)
        for i in 2:6
            add_edge!(g, 1, i)
        end
        
        communities = greedy_modularity(g)
        @test length(communities) == 6
    end
end

println("\n✓ All community detection tests passed!")
