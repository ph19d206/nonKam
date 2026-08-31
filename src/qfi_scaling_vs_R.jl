# src/qfi_scaling_vs_R.jl

"""
    qfi_scaling_vs_R(N::Int, ω::Real, R_array::AbstractVector{<:Real}, K::Real, ϵ::Real, tmax::Int, state;
                     qfi_floor::Float64 = 1e-10, 
                     tol::Float64 = 1e-3,
                     min_points::Int = 10) -> NamedTuple

Calculates the scaling exponent `m` and its standard error `m_err` across a range 
of resonance ratios `R` (where τ = 2π / (ω * R)) using iterative early-time trimming.

# Keyword Arguments
- `qfi_floor::Float64 = 1e-10`: Minimum valid QFI threshold to avoid log(0) divergence.
- `tol::Float64 = 1e-3`: Convergence tolerance for slope stability across trimming steps.
- `min_points::Int = 10`: Minimum number of data points remaining required for a linear fit.

# Returns
A NamedTuple containing:
- `R_array`: Input vector of R values.
- `m_values`: Extracted scaling exponents m(R).
- `m_errors`: Corresponding total standard errors Δm(R).
"""
function qfi_scaling_vs_R(
    N::Int, 
    ω::Real, 
    R_array::AbstractVector{<:Real}, 
    K::Real, 
    ϵ::Real, 
    tmax::Int, 
    state;
    qfi_floor::Float64 = 1e-10,
    tol::Float64 = 1e-3,
    min_points::Int = 10
)
    nR = length(R_array)
    m_values = Vector{Float64}(undef, nR)
    m_errors = Vector{Float64}(undef, nR)

    for (i, R) in enumerate(R_array)
        iszero(R) && error("R cannot be zero (division by zero in τ = 2π / (ω * R)).")
        
        # Calculate time parameter τ for the given R
        τ = 2 * π / (ω * R)
        # println("Calculating for R = $R, τ = $τ")
        
        # Updated call forwarding the new iterative trimming parameters
        res = qfi_scaling_exponent(
            N, ω, τ, K, ϵ, tmax, state; 
            qfi_floor = qfi_floor, 
            tol = tol, 
            min_points = min_points
        )
        
        m_values[i] = res.m
        m_errors[i] = res.m_err
    end

    return (
        R_array = R_array,
        m_values = m_values,
        m_errors = m_errors
    )
end