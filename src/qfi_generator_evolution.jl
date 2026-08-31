# src/qfi_generator_evolution.jl

"""
    qfi_generator_evolution(N::Int, ω::Real, K::Real, τ::Real, t_max::Int, ψ_init::Vector{ComplexF64})

Computes the accumulated Quantum Fisher Information (QFI) time series over `t_max` timesteps.

# Mathematical Definition:
```math
\\text{QFI}(t) = \\sum_{j = 0}^{t-1} \\left[ \\tau \\langle \\psi(j)| \\hat{N} |\\psi(j)\\rangle - \\frac{K}{2\\omega} \\langle \\psi(j)| A \\sin(A) |\\psi(j)\\rangle \\right]
```
"""
function qfi_generator_evolution(
    N::Int, 
    ω::Real, 
    K::Real, 
    τ::Real, 
    t_max::Int, 
    ψ_init::Vector{ComplexF64}
)
    # 1. Pre-allocate QFI output array
    qfi_array = zeros(Float64, t_max)

    # 2. Precompute Asin(A) operator spectrum & eigenvectors (Static over time)
    #    A = (exp(-i*ω*τ)*a + exp(i*ω*τ)*a†) / sqrt(2ω)
    λAsinAop, V_AsinA = AsinAop(N, ω, τ) # Use the fast SymTridiagonal variant
    c_AsinA = (K / (2 * ω)) .* λAsinAop       # Pre-scale eigenvalues by K/(2ω)

    # 3. Precompute Full Single-Step Evolution Operator: U_step = U_kick * U_free
    #    U_kick = V * Diagonal(λ_kick) * V'
    λ_kick, V_kick = eigv_kikU(N, ω, K)
    U_free = free_U(N, ω, τ)                  # Diagonal matrix
    D_kick = Diagonal(λ_kick) # Diagonal matrix for kick operator
    

    # Work buffers for state vector propagation
    buf1 = zeros(ComplexF64, N)
    buf2 = zeros(ComplexF64, N)            # Work buffer for Asin(A) eigenbasis state


    # 4. State vectors and pre-allocated buffer for zero-allocation loop
    ψ = copy(ψ_init)                          # Current state |ψ(t)⟩
    ψ_next = similar(ψ)                       # Work buffer for next state |ψ(t+1)⟩
    ψ_tilde = zeros(ComplexF64, N)            # Work buffer for Asin(A) eigenbasis state

    # Number operator diagonal vector: n = [0, 1, 2, ..., N-1]
    n_diag = 0:(N - 1)

    # 5. Time Evolution Loop
    for t in 1:t_max
        # --- QFI Expectation Value Calculation ---
        
        # A) Term 1: τ * ⟨ψ| N |ψ⟩
        #    sum(n * |ψ_n|^2) avoids matrix-vector multiplication entirely! O(N)
        n_exp = τ * sum(n_diag[k] * abs2(ψ[k]) for k in 1:N)

        # B) Term 2: (K/2ω) * ⟨ψ| A sin(A) |ψ⟩
        #    Transform state to A's eigenbasis: |ψ̃⟩ = V_AsinA' * |ψ⟩
        mul!(ψ_tilde, V_AsinA', ψ)
        
        #    Compute ⟨ψ̃| c_AsinA |ψ̃⟩ = ∑ c_k |ψ̃_k|^2
        AsinA_exp = sum(c_AsinA[k] * abs2(ψ_tilde[k]) for k in 1:N)

        # C) Store QFI
        qfi_array[t] = n_exp - AsinA_exp
        if t > 1
            qfi_array[t] += qfi_array[t - 1] 
        end

        # --- Time Step Propagation (4N² operations, 0 allocations) ---
        mul!(buf1, U_free, ψ)        # 1. Apply free propagation: O(N)
        mul!(buf2, V_kick', buf1)    # 2. Transform to kick basis: O(N²)
        mul!(buf1, D_kick, buf2)     # 3. Apply phase kick: O(N)
        mul!(ψ, V_kick, buf1)        # 4. Transform back to Fock basis: O(N²)
    end

    return qfi_array
end