"""
    grid_around_integer(R0::Int, delta::Real, n_points::Int) -> Vector{Float64}

Generates a symmetric array centered at integer `R0` spanning [R0 - delta, R0 + delta].

# Arguments
- `R0`: Center integer (e.g., 2).
- `delta`: Half-width of the range (e.g., 0.1 gives range [1.9, 2.1]).
- `n_points`: Total number of points. **Must be an odd integer** so R0 is strictly in the middle.
"""
function grid_around_integer(R0::Int, delta::Real, n_points::Int)
    if iseven(n_points)
        n_points += 1  # Force odd number of points so the exact center R0 is included
        @info "Adjusted n_points to $n_points to guarantee the exact center point $R0 is included."
    end
    
    return collect(range(R0 - delta, R0 + delta, length=n_points))
end