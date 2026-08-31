"""
    grid_with_integers(R_min::Real, R_max::Real, n_points::Int) -> Vector{Float64}

Generates a grid from `R_min` to `R_max` with `n_points` total points.
Guarantees that all integer values inside [R_min, R_max] are included.

# Arguments
- `R_min`: Start of the range (e.g., 0.5).
- `R_max`: End of the range (e.g., 3.5).
- `n_points`: Total number of desired points.
"""
function grid_with_integers(R_min::Real, R_max::Real, n_points::Int)
    # 1. Identify all integer boundaries within the range
    int_start = ceil(Int, R_min)
    int_end   = floor(Int, R_max)
    integers  = collect(int_start:int_end)

    # 2. Collect key anchor points: endpoints + integers
    anchors = unique(sort(vcat(Float64(R_min), Float64.(integers), Float64(R_max))))
    
    if n_points <= length(anchors)
        @warn "Requested n_points ($n_points) is <= required integer anchors ($(length(anchors))). Returning anchor points."
        return anchors
    end

    # 3. Subdivide intervals evenly to reach requested n_points
    n_intervals = length(anchors) - 1
    extra_points_needed = n_points - length(anchors)
    
    # Base number of sub-points per interval
    pts_per_interval = fill(div(extra_points_needed, n_intervals), n_intervals)
    
    # Distribute remaining points evenly across middle intervals
    for i in 1:mod(extra_points_needed, n_intervals)
        pts_per_interval[i] += 1
    end

    # 4. Construct final grid
    grid = Float64[anchors[1]]
    for i in 1:n_intervals
        a, b = anchors[i], anchors[i+1]
        n_sub = pts_per_interval[i]
        
        # Generate interior points for interval (a, b)
        sub_pts = range(a, b, length = n_sub + 2)[2:end]
        append!(grid, sub_pts)
    end

    return round.(grid, digits=6)
end