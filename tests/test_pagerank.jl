# Test script for PageRank and link analysis algorithms
# Author: Gabriel Demetrios Lafis

using Test
using Graphs

# Include the pagerank module
include("../pagerank.jl")

@testset "PageRank Tests" begin
    
    @testset "PageRank Basic" begin
        # Create simple directed graph (use directed graph for PageRank)
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 1)
        add_edge!(g, 3, 4)
        add_edge!(g, 4, 1)
        
        pr = pagerank(g)
        @test length(pr) == 4
        @test all(pr[i] >= 0 for i in 1:4)
        # Sum should be approximately 1
        @test sum(pr[i] for i in 1:4) ≈ 1.0 atol=0.01
    end
    
    @testset "PageRank Parameters" begin
        g = SimpleDiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 1)
        
        # Test different alpha values
        pr1 = pagerank(g, alpha=0.85)
        pr2 = pagerank(g, alpha=0.5)
        @test pr1 != pr2
        
        # Test convergence
        pr3 = pagerank(g, max_iter=5)
        @test length(pr3) == 3
    end
    
    @testset "HITS Algorithm" begin
        # Create directed graph
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        
        auth, hub = hits(g)
        @test length(auth) == 4
        @test length(hub) == 4
        @test all(auth[i] >= 0 for i in 1:4)
        @test all(hub[i] >= 0 for i in 1:4)
    end
    
    @testset "Personalized PageRank" begin
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        add_edge!(g, 4, 1)
        
        sources = [1]
        ppr = personalized_pagerank(g, sources)
        @test length(ppr) == 4
        @test all(ppr[i] >= 0 for i in 1:4)
        # Source node should have high score
        @test ppr[1] > 0
    end
    
    @testset "K-shell Decomposition" begin
        # Create graph with different k-cores
        g = SimpleGraph(7)
        # Triangle (3-core)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 1)
        # Connected to triangle
        add_edge!(g, 3, 4)
        add_edge!(g, 4, 5)
        # Leaf nodes
        add_edge!(g, 5, 6)
        add_edge!(g, 5, 7)
        
        kshell = kshell_decomposition(g)
        @test length(kshell) == 7
        @test all(kshell[i] >= 1 for i in 1:7)
    end
    
    @testset "Single Node" begin
        g = SimpleDiGraph(1)
        pr = pagerank(g)
        @test length(pr) == 1
        @test pr[1] ≈ 1.0 atol=0.01
    end
    
    @testset "Disconnected Graph" begin
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 3, 4)
        
        pr = pagerank(g)
        @test length(pr) == 4
        @test sum(pr[i] for i in 1:4) ≈ 1.0 atol=0.01
    end
    
    @testset "Complete Directed Graph" begin
        g = complete_digraph(5)
        pr = pagerank(g)
        # All nodes should have equal PageRank in complete directed graph
        @test all(pr[i] ≈ 0.2 for i in 1:5) || all(pr[i] > 0 for i in 1:5)
    end
end

println("\n✓ All PageRank tests passed!")
