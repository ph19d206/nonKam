"""
    var_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, state)

Calculate the time-dependent variance of the excitation-number operator,
``Var(n̂) = ⟨n̂²⟩ - ⟨n̂⟩²``, for a system evolving up to `t_max` time steps.

# Arguments
- `N::Int`: Dimension of the Hilbert space.
- `ω::Real`, `τ::Real`, `K::Real`: Physical parameters for free evolution and kicking operator.
- `t_max::Int`: Number of evolution time steps.
- `state`: Initial system state. Dispatched on:
  - `Int`: Number state index n in [0, N-1].
  - `Tuple{<:Real, <:Real}`: Coherent state phase space coordinates (x_0, p_0).

# Returns
- `Vector{Float64}`: Array of length `t_max` containing `Var(n̂)` at each time step.
"""
function var_n end

# --- Dispatch 1: Number State |n⟩ ---
function var_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, n::Int)
    0 <= n < N || throw(ArgumentError("Fock state index n=$n must be in range 0 <= n < $N"))

    ψ0 = zeros(ComplexF64, N)
    ψ0[n + 1] = 1.0  # 1-based indexing adjustment

    return _calculate_var_n(N, ω, τ, K, t_max, ψ0)
end

# --- Dispatch 2: Coherent State |x0, p0⟩ ---
function var_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, coherent_coord::Tuple{<:Real, <:Real})
    x0, p0 = coherent_coord
    ψ0 = coherent_state(x0, p0, ω, N)

    return _calculate_var_n(N, ω, τ, K, t_max, ψ0)
end


# ==============================================================================
# Internal Computational Engine
# ==============================================================================

function _calculate_var_n(N::Int, ω::Real, τ::Real, K::Real, t_max::Int, ψ0::Vector{ComplexF64})
    # 1. Operators pre-computation
    λ, Vkick = eigv_kikU(N, ω, K)
    Ufree    = free_U(N, ω, τ)
    D_kick   = Diagonal(λ)

    # 2. Allocate state-vector copy and workspace buffers
    ψ    = copy(ψ0)
    buf1 = similar(ψ)
    buf2 = similar(ψ)

    # 3. Output array initialization
    var_n_vals = Vector{Float64}(undef, t_max)

    # 4. Evolution loop
    for t in 1:t_max
        # Var(n̂) = ⟨n̂²⟩ - ⟨n̂⟩², evaluated directly in the number basis.
        mean, second_moment = _n_moments(ψ)
        var_n_vals[t] = max(0.0, second_moment - mean^2)

        # In-place time step update: ψ(t) -> ψ(t+1)
        evolve!(Ufree, D_kick, Vkick, ψ, buf1, buf2)
    end

    return var_n_vals
end

"""
    _n_moments(ψ)

Compute `⟨n̂⟩` and `⟨n̂²⟩` in one pass, without allocating an operator or
intermediate probability vector.
"""
function _n_moments(ψ::Vector{ComplexF64})
    mean = 0.0
    second_moment = 0.0
    @inbounds for k in 1:length(ψ)
        n = k - 1
        probability = abs2(ψ[k])
        mean += n * probability
        second_moment += n^2 * probability
    end
    return mean, second_moment
end
