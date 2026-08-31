# src/AsinAop.jl

"""
    AsinAop(N::Int, ω::Real, τ::Real) -> (λAsinAop, V_complex)

Computes eigenvalues and phase-corrected eigenvectors for the operator \$A \\sin(A)\$
using real `SymTridiagonal` eigensolvers and post-hoc phase rotation.

See the [Gauge Transformation Theory](@ref) section for the mathematical proof 
showing why this transformation is exact.
"""
function AsinAop(N::Int, ω::Real, τ::Real)
    # Real tridiagonal (phase temporarily removed)
    d = zeros(Float64, N)
    e = sqrt.(1:N-1) ./ sqrt(2 * ω)
    
    # Real SymTridiagonal triggers ultra-fast LAPACK routines (dsterf/dstev)
    λa, V_real = eigen(SymTridiagonal(d, e))

    # Apply phase factor e^(-i * n * ω * τ) to rows of the eigenvector matrix
    phases = exp.(-im * (0:N-1) .* (ω * τ))
    V_complex = phases .* V_real

    λAsinAop = λa .* sin.(λa)

    return λAsinAop, V_complex
end