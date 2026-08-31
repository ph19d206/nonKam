# src/measurement/quadrature_cfi.jl
"""
    quadrature_cfi(ψ_plus, ψ_minus, δ, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ;
                   norm_tol=1e-6, zero_thresh=0.0)

Classical Fisher information of the position-quadrature measurement, estimated
by central finite differences of the initial state:

    CFI(ϕ, θ) = Σ_{x_i} 1/P(x_i, ϕ | θ) · (∂P(x_i, ϕ | θ)/∂θ)²

where ``P(x, ϕ|θ)`` is the rotated-quadrature distribution produced by
`quadrature_distribution` and ``∂P/∂θ`` its finite-difference derivative from
`quadrature_derivative`.  The measurement is optimised over the rotation angle
``ϕ``:

    CFI(θ) = max_ϕ CFI(ϕ, θ)

# Arguments
- `ψ_plus::AbstractVector{<:Complex}`:  state at ``θ+δ``.
- `ψ_minus::AbstractVector{<:Complex}`: state at ``θ-δ``.
- `δ::Real`: central step — forwarded to `quadrature_derivative`.
- `xmin, xmax, Nx`: position grid — forwarded to `quadrature_derivative`.
- `ϕmin, ϕmax, Nϕ`: rotation grid — forwarded to `quadrature_derivative`.
- `norm_tol::Real=1e-6`: normalization tolerance, forwarded to the distribution.
- `zero_thresh::Real=0.0`: saddle over probability bins with `P < zero_thresh`
  (they contribute ``0`` to the sum — the limit ``(∂P)²/P → 0``).

# Returns
A `NamedTuple`
```
(ϕgrid = ϕ         ::Vector{Float64},   # rotation grid
 CFI   = CFI       ::Vector{Float64},   # CFI(ϕ, θ), one per rotation angle
 CFI_opt = maximum(CFI))                # CFI(θ) optimised over ϕ
```
"""
function quadrature_cfi(ψ_plus::AbstractVector{<:Complex},
                        ψ_minus::AbstractVector{<:Complex},
                        δ::Real, xmin::Real, xmax::Real, Nx::Int,
                        ϕmin::Real, ϕmax::Real, Nϕ::Int;
                        norm_tol::Real=1e-6, zero_thresh::Real=0.0)
    # Finite-difference derivative shares a single eigv_xop diagonalisation.
    result = quadrature_derivative(
        ψ_plus, ψ_minus, δ, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ; norm_tol)

    Pgrid = quadrature_distribution(
        (ψ_plus .+ ψ_minus) ./ 2,   # P at the midpoint θ (O(δ²) consistent)
        xmin, xmax, Nx, ϕmin, ϕmax, Nϕ; norm_tol).P

    dP = result.dP   # (Nx, Nϕ)

    # CFI[ϕ] = Σ_{x} (∂P/∂θ)² / P, dropping the P ≈ 0 bins.
    CFI = Vector{Float64}(undef, Nϕ)
    for j in 1:Nϕ
        num = @view dP[:, j]
        den = @view Pgrid[:, j]
        s = 0.0
        for i in eachindex(num)
            den[i] > zero_thresh || continue
            s += num[i]^2 / den[i]
        end
        CFI[j] = s
    end

    return (ϕgrid = result.ϕgrid, CFI = CFI, CFI_opt = maximum(CFI))
end