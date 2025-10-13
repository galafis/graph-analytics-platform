# Test script for centrality algorithms
# Author: Gabriel Demetrios Lafis

using Test
using Graphs

# Include the centrality module
include("../centrality.jl")

@testset "Centrality Tests" begin
    # Create a simple test graph (triangle)
    g = SimpleGraph(4)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 1)
    add_edge!(g, 3, 4)
    
    @testset "Degree Centrality" begin
        dc = degree_centrality(g)
        @test length(dc) == 4
        @test all(0 <= dc[i] <= 1 for i in 1:4)
        # Node 3 should have highest degree centrality (3 connections)
        @test dc[3] > dc[1]
        @test dc[3] > dc[2]
        @test dc[3] > dc[4]
    end
    
    @testset "Betweenness Centrality" begin
        bc = betweenness_centrality(g)
        @test length(bc) == 4
        @test all(bc[i] >= 0 for i in 1:4)
        # Node 3 is a bridge to node 4
        @test bc[3] > 0
    end
    
    @testset "Closeness Centrality" begin
        cc = closeness_centrality(g)
        @test length(cc) == 4
        @test all(0 <= cc[i] <= 1 for i in 1:4)
    end
    
    @testset "Eigenvector Centrality" begin
        ec = eigenvector_centrality(g)
        @test length(ec) == 4
        @test all(ec[i] >= 0 for i in 1:4)
        # Sum should be normalized
        @test sum(ec[i] for i in 1:4) ≈ 1.0 atol=0.01
    end
    
    @testset "Empty Graph" begin
        g_empty = SimpleGraph(0)
        dc = degree_centrality(g_empty)
        @test length(dc) == 0
    end
    
    @testset "Single Node" begin
        g_single = SimpleGraph(1)
        dc = degree_centrality(g_single)
        @test length(dc) == 1
        @test dc[1] == 0.0
    end
    
    @testset "Star Graph" begin
        # Create star graph with center node
        g_star = SimpleGraph(5)
        for i in 2:5
            add_edge!(g_star, 1, i)
        end
        
        dc = degree_centrality(g_star)
        # Center node should have highest degree centrality
        @test dc[1] == 1.0
        @test all(dc[i] == 0.25 for i in 2:5)
    end
    
    @testset "Complete Graph" begin
        g_complete = complete_graph(5)
        dc = degree_centrality(g_complete)
        # All nodes should have same centrality in complete graph
        @test all(dc[i] ≈ 1.0 for i in 1:5)
    end
end

println("\n✓ All centrality tests passed!")
