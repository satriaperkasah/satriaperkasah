# 3. create fd graph from data
module  Horizon
    
using CSV
using DataFrames
using DataFramesMeta
using Graphs
using SimpleWeightedGraphs
using GLMakie
using GraphMakie
using NetworkLayout
using Colors
using StatsBase

export clean_data_set

struct FD
    lhs
    rhs
end

#=
Reads functional dependencies from a file.
Each line in the file represents an FD in the format "LHS -> RHS".
=#
function read_fds(fd_path)
    fds = FD[]

    for line in readlines(fd_path)
        # Skip empty/commented lines
        isempty(strip(line)) && continue
        startswith(line, "#") && continue
        
        # Support both -> and → syntax
        if occursin("->", line)
            parts = split(line, "->")
        else
            @warn "Skipping invalid FD line: $line"
            continue
        end
        
        # Verify format
        length(parts) < 2 && continue
        lhs = strip(parts[1])
        rhs = strip(parts[2])
        
        (isempty(lhs) || isempty(rhs)) && continue
        push!(fds, FD(lhs, rhs))
    end

    return fds
end

#=
Creates mappings between unique attribute values and integer IDs.
This is used to represent nodes in the graph.
=#
function id_map_from_values(values...)
    id_map = Dict{String,Int}()
    keys = unique(sort(reduce(vcat, collect(values))))
    map(x -> id_map[x[2]] = x[1], enumerate(keys))
    reverse_map = Dict{Int,String}(map(reverse, collect(id_map)))
    
    return id_map, reverse_map
end

#=
Section 3.2: FD Pattern Graph Contruction
Builds the FD pattern graph (FDG) from data and functional dependencies.
The FDG represents FD patterns as edges, weighted by their support in the data.
=#
function build_fd_graph(data, fds)
    # prebuild nodes and edges in list to improve graph creation: https://juliagraphs.org/SimpleWeightedGraphs.jl/dev/#Caveats
    source_list = Vector{String}() 
    dest_list = Vector{String}()
    weights = Vector{Int}()
    edge_to_fd = Dict{Tuple{Int,Int},FD}()  # Track FD for each edge

    # Create normalized column map
    norm_cols = lowercase.(replace.(names(data), r"\s+" => "_"))
    col_map = Dict(norm_cols .=> names(data))

    for fd in fds
        # Normalize FD attributes
        norm_lhs = lowercase(replace(fd.lhs, r"\s+" => "_"))
        norm_rhs = lowercase(replace(fd.rhs, r"\s+" => "_"))

        # Find original column names
        orig_lhs = get(col_map, norm_lhs, nothing)
        orig_rhs = get(col_map, norm_rhs, nothing)

        # Check if columns exist
        if orig_lhs === nothing || orig_rhs === nothing
            missing_cols = String[]
            orig_lhs === nothing && push!(missing_cols, fd.lhs)
            orig_rhs === nothing && push!(missing_cols, fd.rhs)
            @warn "Skipping FD: $(fd.lhs) → $(fd.rhs). Columns not found: $(join(missing_cols, ", "))"
            continue
        end

        if hasproperty(data, Symbol(orig_lhs)) && hasproperty(data, Symbol(orig_rhs))
            # Add all nodes from LHS and RHS of FD to node_list and add prefix to have only unique values per column
            fd_source_list = combine(groupby(data, [Symbol(orig_lhs), Symbol(orig_rhs)]), nrow => :count)
            for row in eachrow(fd_source_list)
                # Create nodes with "attribute:value" format
                lhs_node = "$(orig_lhs):$(row[Symbol(orig_lhs)])"
                rhs_node = "$(orig_rhs):$(row[Symbol(orig_rhs)])"
                
                # Add edge with weight = support count
                push!(source_list, lhs_node)
                push!(dest_list, rhs_node)
                push!(weights, row[:count])
            end
        else
            @warn "Column(s) '$orig_lhs' or '$orig_rhs' not accessible in DataFrame for dependency '$fd'."
        end
    end

    # Handle empty graph case
    if isempty(source_list)
        @warn "No valid FD patterns found. Returning empty graph."
        return SimpleWeightedDiGraph(), Dict{String,Int}(), Dict{Int,String}()
    end

    value_to_id, id_to_value = id_map_from_values(source_list, dest_list) # create map and reverse map to translate between values and graph ids
    weighted_di_graph = SimpleWeightedDiGraph([value_to_id[x] for x in source_list], [value_to_id[x] for x in dest_list],weights) # Construct the weighted directed graph
    
     # Build edge_to_fd mapping
     for (i, (s, d)) in enumerate(zip(source_list, dest_list))
        u = value_to_id[s]
        v = value_to_id[d]
        # Extract attribute from node string
        lhs_attr = split(s, ":")[1]
        rhs_attr = split(d, ":")[1]
        # Find matching FD
        for fd in fds
            if fd.lhs == lhs_attr && fd.rhs == rhs_attr
                edge_to_fd[(u, v)] = fd
                break
            end
        end
    end
    
    # Create graph with weighted edges
    return weighted_di_graph, value_to_id, id_to_value, edge_to_fd
end

# 1305 - Section 3.3: Pattern Quality
function compute_pattern_quality(FDG::SimpleWeightedDiGraph, id_to_value::Dict, edge_to_fd::Dict, lhs_group_sizes::Dict, is_cyclic::Bool)
    edge_quality = Dict{Tuple{Int,Int}, Float64}()
    vertex_quality = Dict{Int,Float64}()
    
    # 1. Compute conditional support for each edge (pattern)
    edge_support = Dict{Tuple{Int64, Int64},Float64}([((src(e), dst(e)), get_weight(FDG, e) / ne(FDG)) for e in edges(FDG)]) # Equation 1
    vertex_quality = Dict{Int,Float64}()
    back_edges = Set{Tuple{Int,Int}}()
    visited = falses(nv(FDG))

    for v in vertices(FDG)
        visited[v] = true
        neighbor_count = 0
        neighbor_quality = 0
        for w in neighbors(FDG, v)
            if visited[w]
                push!(back_edges, (v, w))
            else
                derived_fds_support = map(x -> edge_support[(src(x), dst(x))], edges(dfs_tree(FDG, w))) # calculate support for all following fds 
                edge_quality[(v,w)] = (edge_support[(v,w)] + sum(derived_fds_support)) / (1 + length(derived_fds_support)) # Equation 2
                # accumulate edge qualities to determine vertex quality later
                neighbor_count += 1 
                neighbor_quality += edge_quality[(v,w)]
            end
        end
        vertex_quality[v] = neighbor_quality / neighbor_count
    end

    for (head, tail) in back_edges
        edge_quality[(head, tail)] = vertex_quality[tail]
    end

    for ((l,r), v) in edge_quality
        println("$(split(id_to_value[l], ':')[2]) -> $(split(id_to_value[r], ':')[2]): $v   ")
    end
    return edge_quality
end

# 1305 - Visualize FDG
function visualize_fdg(FDG, id_to_value, edge_quality::Dict; title="FD Pattern Graph", node_size=15, edge_width_scale=5)
    fig = Figure(size=(800, 600), fontsize=12)
    ax = Axis(fig[1,1], title=title)

    # Node labels (assuming format "ID:Description")
    node_labels = [split(id_to_value[i], ":")[2] for i in 1:nv(FDG)]

    # Convert edge qualities to array in edge order
    edge_qualities = [edge_quality[(src(e), dst(e))] for e in edges(FDG)]
    edge_widths = edge_width_scale .* (edge_qualities ./ maximum(edge_qualities))

    # Layout algorithm (Spring/Stress/Shell)
    layout = NetworkLayout.Spring(dim=2)

    # Plot the graph with styled edges and nodes
    graphplot!(ax, FDG,
        layout=layout,
        nlabels=node_labels,
        nlabels_align=(:center, :center),
        nlabels_distance=5,
        node_size=node_size,
        node_color=colorant"lightblue",
        edge_color=[eq > 0.9*maximum(edge_qualities) ? colorant"green" : colorant"red" for eq in edge_qualities],
        edge_width=edge_widths,
        arrow_size=15,
        arrow_shift=:end)

    # Add colorbar (use either `limits` OR `colorrange`, not both)
    Colorbar(fig[1,2], 
        colormap=[colorant"red", colorant"green"],
        colorrange=extrema(edge_qualities),  # Replaced `limits` with `colorrange`
        label="Edge Quality")

    # Clean up axis
    hidespines!(ax)
    hidedecorations!(ax)

    display(fig)
    return fig
end

# 1805 - Section 4: Rules Compilation
function analyze_fd_interactions(fds::Vector{FD})
    isempty(fds) && return (
        bound_attrs = Set{String}(),
        free_attrs = Set{String}(),
        processing_order = Int[],
        is_cyclic = false
    )
    
    # Identify bound/free attributes (Section 4.2)
    lhs_attrs = Set{String}([fd.lhs for fd in fds])
    free_attrs = Vector{String}([fd.rhs for fd in fds]) # -> all attributes that are on the rhs of a FD
    bound_attrs = setdiff(lhs_attrs, free_attrs)
    
    # Build interaction graph for Cases 1-4
    g = SimpleDiGraph(length(fds))
    for (i, fd_i) in enumerate(fds), (j, fd_j) in enumerate(fds)
        i == j && continue

        # Determine interaction case
        if fd_i.lhs == fd_j.lhs && fd_i.rhs != fd_j.rhs
            add_edge!(g, i, j)  # Case 1 A→B, A→C
        elseif fd_i.rhs == fd_j.rhs
            add_edge!(g, i, j)  # Case 2 # A→C, B→C
            add_edge!(g, j, i)  # bidirectional
        elseif fd_i.rhs == fd_j.lhs
            add_edge!(g, i, j)  # Case 3 # A→B, B→C
        elseif fd_i.lhs == fd_j.rhs && fd_i.rhs == fd_j.lhs
            add_edge!(g, i, j)  # Case 4 # A→B, B→A (cyclic)
            add_edge!(g, j, i)
        end
    end
    
    # Compute processing order using SCCs (Section 5.1)
    scc = strongly_connected_components(g)
    comp_order = topological_sort(SimpleDiGraph(length(scc)))
    processing_order = Int[]
    for comp_idx in comp_order
        append!(processing_order, scc[comp_idx])
    end
    
    return (
        bound_attrs = bound_attrs,
        free_attrs = free_attrs,
        processing_order = processing_order,
        is_cyclic = any(length(c) > 1 for c in scc)
    )
end

# 2405 - Section 5: Traversing the FD Pattern Graph
function repair_data(data::DataFrame, FDG::SimpleWeightedDiGraph, fds::Vector{FD}, value_to_id::Dict, id_to_value::Dict, edge_quality::Dict, analysis::NamedTuple)
    
    repaired = data
    repair_tables = Dict{String,Dict{String,String}}()
    pattern_expressions = Dict{Int,Vector{Tuple{FD,String}}}()
    
    # Initialize repair tables
    for fd in fds
        repair_tables["$(fd.lhs)->$(fd.rhs)"] = Dict{String,String}()
    end
    
    # Initialize pattern expressions
    for idx in 1:nrow(repaired)
        pattern_expressions[idx] = Tuple{FD,String}[]
    end
    
    # Repair each tuple
    for (idx, row) in enumerate(eachrow(repaired))
        # Track processed FDs
        processed_fds = Set{Int}()
        
        # Process FDs in order
        for fd_idx in analysis.processing_order
            fd = fds[fd_idx]
            fd_key = "$(fd.lhs)->$(fd.rhs)"
            fd_idx in processed_fds && continue
            
            lhs_val = string(row[Symbol(fd.lhs)])
            current_rhs_val = string(row[Symbol(fd.rhs)])
            lhs_node = "$(fd.lhs):$lhs_val"
            
            # Check cache first
            if haskey(repair_tables[fd_key], lhs_val)
                cached_val = repair_tables[fd_key][lhs_val]
                repaired[idx, Symbol(fd.rhs)] = cached_val
                push!(pattern_expressions[idx], (fd, "REPAIRED: $cached_val (cached)"))
                push!(processed_fds, fd_idx)
                continue
            end
            
            # Find best candidate
            best_quality = -1.0
            best_rhs_val = nothing
            
            if haskey(value_to_id, lhs_node)
                lhs_id = value_to_id[lhs_node]

                # Find best candidate using quality scores 
                for v in outneighbors(FDG, lhs_id)
                    node_str = id_to_value[v]
                    if startswith(node_str, "$(fd.rhs):")
                        quality = get(edge_quality, (lhs_id, v), 0.0)
                        if quality > best_quality
                            best_quality = quality
                            best_rhs_val = split(node_str, ":")[2]  # Extract value
                        end
                    end
                end
            end
            
            # Apply repair if valid candidate found
            if best_rhs_val !== nothing && best_rhs_val != current_rhs_val
                repaired[idx, Symbol(fd.rhs)] = best_rhs_val
                repair_tables[fd_key][lhs_val] = best_rhs_val
                push!(pattern_expressions[idx], (fd, "REPAIRED: $best_rhs_val (q=$(round(best_quality, digits=2))"))
            else
                repair_tables[fd_key][lhs_val] = current_rhs_val
                push!(pattern_expressions[idx], (fd, "NO_CHANGE: $current_rhs_val"))
            end
            
            push!(processed_fds, fd_idx)
        end
    end
    
    return repaired, pattern_expressions
end

function clean_data_set(data_path, fd_path, output_path, visualize = true)
    # 1. Read data and FDs
    data = CSV.read(data_path, DataFrame)
    fds = read_fds(fd_path)
    
    # Convert FD columns to String type
    for fd in fds
        for col in [fd.lhs, fd.rhs]
            if hasproperty(data, Symbol(col)) && !(eltype(data[!, Symbol(col)]) <: AbstractString)
                data[!, Symbol(col)] = string.(data[!, Symbol(col)])
            end
        end
    end

    # Precompute LHS group sizes for each FD
    lhs_group_sizes = Dict{Tuple{String,String}, Dict{String,Int}}()
    for fd in fds
        if hasproperty(data, Symbol(fd.lhs))
            group_sizes = combine(groupby(data, Symbol(fd.lhs)), nrow => :count)
            lhs_dict = Dict{String,Int}()
            for row in eachrow(group_sizes)
                lhs_dict[string(row[1])] = row[:count]
            end
            lhs_group_sizes[(fd.lhs, fd.rhs)] = lhs_dict
        end
    end

    # 2. Build FD graph (Section 3.2)
    FDG, value_to_id, id_to_value, edge_to_fd = build_fd_graph(data, fds)
    
    # 4. Analyze FD interactions (Section 4)
    analysis = analyze_fd_interactions(fds)
    
    # 3. Compute pattern quality (Section 3.3)
    edge_quality = compute_pattern_quality(FDG, id_to_value, edge_to_fd, lhs_group_sizes, analysis.is_cyclic)

    fig_before = nothing
    if visualize
        fig_before = visualize_fdg(FDG, id_to_value, edge_quality, title="Original FD Pattern Graph $(analysis.is_cyclic ? "(Cyclic)" : "")")
    end
    # TODO: save figure to file

    # 5. Repair data (Section 5)
    repaired_data, pattern_expressions = repair_data(data, FDG, fds, value_to_id, id_to_value, edge_quality, analysis)
    
    # 6. Save results
    CSV.write(output_path, repaired_data)

    open("repair_report.txt", "w") do io
        println(io, "Data Repair Report")
        println(io, "="^50)
        for (idx, exprs) in pattern_expressions
            println(io, "\nTuple $idx:")
            for (fd, action) in exprs
                println(io, "  • $(fd.lhs) → $(fd.rhs) : $action")
            end
        end
    end
    return 
end
end # end of module

# main("testdata/paper.csv", "fd/paper_fd.txt")
# Horizon.clean_data_set("testdata/raha_hospital_dirty.csv", "fd/mock_fd.txt", "output/testresult.csv")
Horizon.clean_data_set("testdata/paper.csv", "fd/paper_fd.txt", "output/testresult.csv")
# Horizon.clean_data_set("testdata/raha_flights_dirty.csv", "fd/flights_fd.txt", "output/testresult.csv")