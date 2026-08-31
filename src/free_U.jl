# free_U.jl
"""
    free_U(N, ω, τ)
Compute N x N Floquet unitary matrix.
"""
function free_U(N, ω, τ)
    # Free evolution: U_free = exp(-i * ω * τ * a^† a)
    n_diag = 0:(N-1)
    exp_diag = exp.(-im * ω * τ .* n_diag)
    U_free = Diagonal(exp_diag)
    return U_free
end