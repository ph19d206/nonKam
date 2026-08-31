# floquet_qfi.jl
using ProgressMeter
"""
    floquet_qfi(N, ω, τ, K, ϵ, t_max, state; progress=false, parallel=true)

Computes the Quantum Fisher Information (QFI) profile over time for a kicked 
harmonic oscillator using numerical derivative tracking of state Fidelity.

The tracking uses the central expansion relation:
    \$F(t) \\approx 1 - \\frac{1}{4} \\epsilon^2 \\text{QFI}(t)\$

# Arguments
- `N::Int`: Dimension of the truncated Hilbert space (number of Fock states).
- `ω::Real`: Unperturbed angular frequency of the harmonic oscillator.
- `τ::Real`: Driving period (time between successive kicks).
- `K::Real`: Kick strength parameter.
- `ϵ::Real`: Infinitesimal perturbation added to the driving frequency (ω + ϵ).
- `t_max::Int`: Total number of discrete Floquet periods (time steps) to simulate.
- `state`: Initial system state. Can be an `Int` (representing a number state index \$n\$) 
  or a `Tuple{Real, Real}` (representing coherent state coordinates \$(x_0, p_0)\$).
- `progress::Bool=false`: Display a progress bar when `true`.
- `parallel::Bool=true`: Evolve the unperturbed and perturbed trajectories on separate
  Julia threads when more than one thread is available.

# Returns
- `Vector{Float64}`: A profile array of size `t_max` containing the evaluated QFI values.
"""
function floquet_qfi end  # Base function declaration for multiple dispatch


# --- Method 1: For Number States (Fock States) ---
"""
    floquet_qfi(N, ω, τ, K, ϵ, t_max, n::Int)

Dispatches the QFI evolution tracking initialized via a pure Fock state \$|n\\rangle\$.
"""
function floquet_qfi(
    N::Int, ω::Real, τ::Real, K::Real, ϵ::Real, t_max::Int, n::Int;
    progress::Bool=false, parallel::Bool=true,
)
    # Bounds check to prevent out-of-bounds structural errors in a finite basis
    if n < 0 || n >= N
        throw(ArgumentError("Fock state index n must be bound within 0 <= n < N (N=$N)."))
    end
    
    # Initialize the initial state vector |ψ(0)⟩ in the Fock basis
    ψ0 = zeros(ComplexF64, N)
    ψ0[n + 1] = 1.0  # +1 adjustment due to Julia's 1-based indexing structure
    
    return _compute_qfi_evolution(N, ω, τ, K, ϵ, t_max, ψ0;
                                  progress=progress, parallel=parallel)
end


# --- Method 2: For Coherent States (Glauber States) ---
"""
    floquet_qfi(N, ω, τ, K, ϵ, t_max, coherent_coord::Tuple{Real, Real})

Dispatches the QFI evolution tracking initialized via an external `coherent_state` generator.
"""
function floquet_qfi(
    N::Int, ω::Real, τ::Real, K::Real, ϵ::Real, t_max::Int,
    coherent_coord::Tuple{Real, Real}; progress::Bool=false, parallel::Bool=true,
)
    x0, p0 = coherent_coord
    
    # Generate the initial coherent state using your pre-built function
    ψ0 = coherent_state(x0, p0, ω, N)
    
    return _compute_qfi_evolution(N, ω, τ, K, ϵ, t_max, ψ0;
                                  progress=progress, parallel=parallel)
end


# --- Core Engine: Fast In-Place Memory-Efficient Time Evolution ---
"""
    _compute_qfi_evolution(N, ω, τ, K, ϵ, t_max, ψ0)

Internal simulation engine. Progresses both unperturbed (\$U_{\\text{free}}(\\omega)\$)
and perturbed (\$U_{\\text{free}}(\\omega + \\epsilon)\$) trajectories. When enabled,
the independent updates run concurrently on Julia threads. Each trajectory uses dedicated,
pre-allocated work buffers and in-place BLAS operations.
"""
function _compute_qfi_evolution(
    N::Int, ω::Real, τ::Real, K::Real, ϵ::Real, t_max::Int,
    ψ0::Vector{ComplexF64}; progress::Bool=false, parallel::Bool=true,
)
    # 1. Fetch pre-constructed component matrices/operators
    λ, V = eigv_kikU(N, ω, K)     # Diagonal eigen-elements of the Kicking Operator
    U_free = free_U(N, ω, τ)         # Structural Diagonal type for unperturbed path
    U_free_ϵ = free_U(N, ω + ϵ, τ)   # Structural Diagonal type for perturbed path (ω + ϵ)
    
    # 2. Wrap Kick eigenvalues into a dedicated structural Diagonal type
    D_kick = Diagonal(λ)
    
    # Pre-allocate state vectors to eliminate runtime allocation jitter inside the loop
    ψ = copy(ψ0)
    ψϵ = copy(ψ0)
    
    # Each path owns its workspace, so the two evolutions can safely run concurrently.
    buf1 = zeros(ComplexF64, N)
    buf2 = zeros(ComplexF64, N)
    buf1_ϵ = zeros(ComplexF64, N)
    buf2_ϵ = zeros(ComplexF64, N)
    
    # Profile container array to map QFI values over time slices
    qfi_profile = zeros(Float64, t_max)

    # Keep terminal output opt-in; this is useful for batch simulations.
    p = progress ? Progress(t_max; dt=0.5, desc="Simulating Floquet QFI Evolution: ") : nothing
    
    # 3. Time Evolution Loop
    for t in 1:t_max
        # The updates share no mutable state.  Keep their work buffers separate and
        # synchronize before calculating the fidelity at this time step.
        if parallel && Threads.nthreads() > 1
            @sync begin
                Threads.@spawn evolve!(U_free, D_kick, V, ψ, buf1, buf2)
                evolve!(U_free_ϵ, D_kick, V, ψϵ, buf1_ϵ, buf2_ϵ)
            end
        else
            evolve!(U_free, D_kick, V, ψ, buf1, buf2)
            evolve!(U_free_ϵ, D_kick, V, ψϵ, buf1_ϵ, buf2_ϵ)
        end
        
        # --- C. Extract Quantum Fisher Information ---
        # Evaluate QFI from the overlap using your pre-built function
        qfi_profile[t] = calculate_qfi_fidility(ψ, ψϵ, ϵ)

        # --- Update Progress Bar ---
        progress && next!(p)
    end
    
    return qfi_profile
end


"""
    floquet_qfi_R_sweep(N, ω, R_values, K, ϵ, t_max, state;
                        progress=false, parallel=true) -> NamedTuple

Calculate QFI time traces for several resonance parameters `R_values`, keeping
`ω` fixed.  For every value of `R`, the kicking period is set to
`τ = 2π / (ω * R)`.

The returned `qfi` matrix has shape `(t_max, length(R_values))`: column `j`
is the QFI trajectory for `R[j]`.  `time_steps` gives the common Floquet-kick
index, while `times[:, j]` gives the corresponding physical time for column
`j` because the period varies with `R`.
"""
function floquet_qfi_R_sweep(
    N::Int, ω::Real, R_values::AbstractVector{<:Real}, K::Real, ϵ::Real,
    t_max::Int, state; progress::Bool=false, parallel::Bool=true,
)
    N > 0 || throw(ArgumentError("N must be positive."))
    t_max > 0 || throw(ArgumentError("t_max must be positive."))
    iszero(ω) && throw(ArgumentError("ω cannot be zero when τ = 2π / (ω * R)."))
    isempty(R_values) && throw(ArgumentError("R_values cannot be empty."))

    R = Float64.(R_values)
    all(isfinite, R) || throw(ArgumentError("Every R value must be finite."))
    all(x -> !iszero(x), R) || throw(ArgumentError("R values cannot be zero."))

    τ = 2π ./ (Float64(ω) .* R)
    qfi = Matrix{Float64}(undef, t_max, length(R))
    time_steps = collect(1:t_max)

    for j in eachindex(R)
        progress && println("QFI sweep: R = $(R[j]) ($(j)/$(length(R)))")
        qfi[:, j] = floquet_qfi(N, ω, τ[j], K, ϵ, t_max, state;
                                 progress=false, parallel=parallel)
    end

    times = Float64.(time_steps) .* permutedims(τ)
    return (R=R, tau=τ, time_steps=time_steps, times=times, qfi=qfi)
end
