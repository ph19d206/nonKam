# KickedHO_floquet

"""
    KickedHO_floquet(N, ω, τ, K)

Constructs the N x N Floquet unitary matrix for the kicked harmonic oscillator.
"""
function KickedHO_floquet(N::Int, ω::Real, τ::Real, K::Real)
    # 1. The X operator. It is a symmetric tridiagonal matrix.
    off_diagonal = sqrt.(1:N-1)
    zero_diagonal = zeros(N)
    Xop = SymTridiagonal(zero_diagonal, off_diagonal)
    λx, V = eigen(Xop)
    λx .= (1 / sqrt(2 * ω)) * λx # as X = (a+a^†)/(sqrt(2*ω))

    # 2. The kick evolution: U_kick = exp(-i * K * cos(X))
    # We apply the function element-wise to the eigenvalues in the diagonal basis
    # and rotate back to the Fock basis using V.
    U_kick_diag = exp.(-im * K .* cos.(λx))
    
    # Reconstruct the full unitary operator in the Fock basis: V * diag * V^T
    # (Since Xop is real and symmetric, its eigenvectors V are real, so V^† = V^T)
    U_kick = V * diagm(U_kick_diag) * V'

    # 3. Free evolution: U_free = exp(-i * ω * τ * a^† a)
    # It is purely diagonal, so we construct it directly for efficiency
    n_diag = 0:(N-1)
    U_free = diagm(exp.(-im * ω * τ .* n_diag))

    # 4. Full Floquet Unitary
    return U_kick * U_free

    # --- new version ---
    # off_diagonal = sqrt.(1:N-1)
    # zero_diagonal = zeros(N)
    # Xop = SymTridiagonal(zero_diagonal, off_diagonal)
    # λx, V = eigen(Xop)
    # λx .= (1 / sqrt(2 * ω)) * λx 

    # U_kick_diag = exp.(-im * K .* cos.(λx))
    # U_kick = (V * Diagonal(U_kick_diag)) * V'  # Swapped diagm for Diagonal

    # n_diag = 0:(N-1)
    # U_free = Diagonal(exp.(-im * ω * τ .* n_diag))  # Swapped diagm for Diagonal

    # return U_kick * U_free
end