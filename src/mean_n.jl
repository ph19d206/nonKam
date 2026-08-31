# src/mean_n.jl
"""
    mean_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, state)

Calculate the time-dependent mean excitation number, for a system evolving up to `t_max` time steps.

# Arguments
- `N::Int`: Dimension of the Hilbert space.
- `ω::Real`, `τ::Real`, `K::Real`: Physical parameters for free evolution and kicking operator.
- `t_max::Int`: Number of evolution time steps.
- `state`: Initial system state. Dispatched on:
  - `Int`: Number state index n in [0, N-1].
  - `Tuple{<:Real, <:Real}`: Coherent state phase space coordinates (x_0, p_0).

# Returns
- `Vector{Float64}`: Array of length `t_max` containing the expectation value at each time step.
"""
function mean_n end 

# --- Dispatch 1: Number State |n⟩ ---
function mean_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, n::Int)
    0 <= n < N || throw(ArgumentError("Fock state index n=$n must be in range 0 <= n < $N"))
    
    ψ0 = zeros(ComplexF64, N)
    ψ0[n + 1] = 1.0  # 1-based indexing adjustment
    
    return _calculate_mean_n(N, ω, τ, K, t_max, ψ0)
end

# --- Dispatch 2: Coherent State |x0, p0⟩ ---
function mean_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, coherent_coord::Tuple{<:Real, <:Real})
    x0, p0 = coherent_coord
    ψ0 = coherent_state(x0, p0, ω, N)
    
    return _calculate_mean_n(N, ω, τ, K, t_max, ψ0)
end


# ==============================================================================
# Internal Computational Engine
# ==============================================================================

function _calculate_mean_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, ψ0::Vector{ComplexF64})
    # 1. Operators pre-computation
    λ, Vkick = eigv_kikU(N, ω, K)
    Ufree    = free_U(N, ω, τ)      # Assumed to return a Diagonal operator
    D_kick   = Diagonal(λ)

    # 2. Allocate state vector copy and workspace buffers (for zero-allocation time loop)
    ψ    = copy(ψ0)
    buf1 = similar(ψ)
    buf2 = similar(ψ)

    # 3. Output array initialization
    mean_n_vals = Vector{Float64}(undef, t_max)

    # 4. Evolution loop
    for t in 1:t_max
        # Record <ψ(t)| n |ψ(t)> via fast zero-allocation reduction
        mean_n_vals[t] = _expectation_n(ψ)

        # In-place time step update: ψ(t) -> ψ(t+1)
        evolve!(Ufree, D_kick, Vkick, ψ, buf1, buf2)
    end

    return mean_n_vals
end

"""
    _expectation_n(ψ)

Compute ⟨ψ|n|ψ⟩ = ∑ₖ k |ψₖ₊₁|² for k ∈ 0:N-1 without extra memory allocations.
"""
function _expectation_n(ψ::Vector{ComplexF64})
    val = 0.0
    @inbounds for k in 1:length(ψ)
        val += (k - 1) * abs2(ψ[k])
    end
    return val
end