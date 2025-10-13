# Citation Network Analysis Example
# Author: Gabriel Demetrios Lafis

using Graphs
using Printf

# Include modules
include("../julia/centrality.jl")
include("../julia/pagerank.jl")
include("../julia/community.jl")

"""
Analyze citation network to identify:
- Influential papers (PageRank)
- Research communities
- Key papers (centrality)
"""
function main()
    println("\n" * "="^60)
    println("CITATION NETWORK ANALYSIS EXAMPLE")
    println("="^60 * "\n")
    
    # 1. Create sample citation network
    println("1. Creating Sample Citation Network...")
    g = SimpleDiGraph(20)
    
    # Add citation edges (directed: paper A cites paper B)
    citations = [
        (1, 5), (1, 6), (2, 5), (2, 7), (3, 5), (3, 8),
        (4, 6), (4, 8), (5, 10), (6, 10), (7, 11), (8, 11),
        (9, 12), (10, 15), (11, 15), (12, 16), (13, 16),
        (14, 17), (15, 18), (16, 18), (17, 19), (18, 20)
    ]
    
    for (src, dst) in citations
        add_edge!(g, src, dst)
    end
    
    println("   Network created: $(nv(g)) papers, $(ne(g)) citations")
    
    # 2. Calculate PageRank to identify influential papers
    println("\n2. Identifying Influential Papers (PageRank)...")
    pr = pagerank(g, alpha=0.85)
    
    # Sort by PageRank
    sorted_papers = sort(collect(pr), by=x->x[2], rev=true)
    println("   Top 10 most influential papers:")
    for i in 1:min(10, length(sorted_papers))
        paper, score = sorted_papers[i]
        @printf("     Paper %2d: %.4f\n", paper, score)
    end
    
    # 3. Calculate HITS (Authority and Hub scores)
    println("\n3. HITS Analysis (Authority & Hub Scores)...")
    auth, hub = hits(g)
    
    # Sort by authority
    sorted_auth = sort(collect(auth), by=x->x[2], rev=true)
    println("   Top 5 Authorities (highly cited papers):")
    for i in 1:min(5, length(sorted_auth))
        paper, score = sorted_auth[i]
        @printf("     Paper %2d: %.4f\n", paper, score)
    end
    
    # Sort by hub
    sorted_hub = sort(collect(hub), by=x->x[2], rev=true)
    println("   Top 5 Hubs (papers that cite many others):")
    for i in 1:min(5, length(sorted_hub))
        paper, score = sorted_hub[i]
        @printf("     Paper %2d: %.4f\n", paper, score)
    end
    
    # 4. Find research communities (convert to undirected for this)
    println("\n4. Detecting Research Communities...")
    g_undirected = SimpleGraph(g)
    communities = label_propagation(g_undirected)
    
    # Group by community
    community_groups = Dict{Int, Vector{Int}}()
    for (paper, comm) in communities
        if !haskey(community_groups, comm)
            community_groups[comm] = Int[]
        end
        push!(community_groups[comm], paper)
    end
    
    println("   $(length(community_groups)) research communities detected:")
    for (comm_id, papers) in sort(collect(community_groups), by=x->length(x[2]), rev=true)
        println("     Community $comm_id: $(length(papers)) papers")
        if length(papers) <= 5
            println("       Papers: $papers")
        end
    end
    
    # 5. Calculate modularity
    Q = modularity(g_undirected, communities)
    @printf("\n   Modularity: %.4f\n", Q)
    
    # 6. Identify key papers using centrality
    println("\n5. Key Papers Analysis (Centrality Measures)...")
    
    # Degree centrality (in-degree = times cited)
    in_degrees = [indegree(g, v) for v in vertices(g)]
    out_degrees = [outdegree(g, v) for v in vertices(g)]
    
    println("   Top 5 most cited papers (in-degree):")
    cited_papers = sort(collect(enumerate(in_degrees)), by=x->x[2], rev=true)
    for i in 1:min(5, length(cited_papers))
        paper, citations = cited_papers[i]
        println("     Paper $paper: $citations citations")
    end
    
    println("\n   Top 5 papers with most references (out-degree):")
    citing_papers = sort(collect(enumerate(out_degrees)), by=x->x[2], rev=true)
    for i in 1:min(5, length(citing_papers))
        paper, refs = citing_papers[i]
        println("     Paper $paper: $refs references")
    end
    
    # 7. K-shell decomposition to find core papers
    println("\n6. K-Shell Decomposition...")
    kshell = kshell_decomposition(g_undirected)
    
    max_shell = maximum(values(kshell))
    println("   Maximum k-shell: $max_shell")
    
    # Papers in highest k-shell
    core_papers = [p for (p, k) in kshell if k == max_shell]
    println("   Core papers (highest k-shell): $core_papers")
    
    # 8. Personalized PageRank for topic-specific ranking
    println("\n7. Personalized PageRank (Topic-Specific Ranking)...")
    seed_papers = [1, 2]  # Starting from early papers
    ppr = personalized_pagerank(g, seed_papers)
    
    sorted_ppr = sort(collect(ppr), by=x->x[2], rev=true)
    println("   Top 5 related papers to seeds $seed_papers:")
    for i in 1:min(5, length(sorted_ppr))
        paper, score = sorted_ppr[i]
        @printf("     Paper %2d: %.4f\n", paper, score)
    end
    
    # 9. Summary
    println("\n" * "="^60)
    println("ANALYSIS SUMMARY")
    println("="^60)
    println("Network: $(nv(g)) papers, $(ne(g)) citations")
    println("Communities: $(length(community_groups))")
    println("Most influential paper: Paper $(sorted_papers[1][1])")
    println("Most cited paper: Paper $(cited_papers[1][1]) ($(cited_papers[1][2]) citations)")
    println("Modularity: $(@sprintf("%.4f", Q))")
    println("="^60 * "\n")
    
    println("✓ Citation network analysis complete!")
end

# Run main function
main()
