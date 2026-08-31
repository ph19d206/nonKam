# src/measurement/quadrature_derivative.jl
"""
    quadrature_derivative(ψ_plus, ψ_minus, δ, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ;
                          norm_tol=1e-6)

Central finite difference of the quadrature distribution with respect to a
parameter ``θ`` that enters the problem through the initial state ``ψ(θ)``:

    ∂P(x, ϕ|θ)     P(x, ϕ|θ+δ)  -  P(x, ϕ|θ-δ)
    ───────────  ≈  ─────────────────────────────
        ∂θ                      2δ

# Arguments
- `ψ_plus::AbstractVector{<:Complex}`:  state prepared at ``θ+δ``.
- `ψ_minus::AbstractVector{<:Complex}`: state prepared at ``θ-δ``.
- `δ::Real`: central step, must be `> 0`. Error is ``O(δ²)``.
- `xmin, xmax, Nx`: position grid — forwarded to `quadrature_distribution`.
- `ϕmin, ϕmax, Nϕ`: rotation grid — forwarded to `quadrature_distribution`.
- `norm_tol::Real=1e-6`: normalization tolerance, forwarded to
  `quadrature_distribution`.

The X eigenbasis is computed **once** and reused for both perturbed states
(the expensive `eigv_xop` diagonalization is not repeated), so only the
projection + interpolation work is done twice per derivative.

# Returns
A `NamedTuple`
```
(xgrid = x          ::Vector{Float64},   # position grid
 ϕgrid = ϕ          ::Vector{Float64},   # rotation grid
 dP    = dP         ::Matrix{Float64})   # ∂P/∂θ, size (Nx, Nϕ)
```
with `dP[i, j] ≈ ∂P(x_i, ϕ_j)/∂θ`.

The classical Fisher information follows from the squares of this matrix:
```julia
CFI(θ) ≈ sum(abs2(dP) ./ P)   # summed over the sampled x quadrature values
```
"""
function quadrature_derivative(ψ_plus::AbstractVector{<:Complex},
                               ψ_minus::AbstractVector{<:Complex},
                               δ::Real, xmin::Real, xmax::Real, Nx::Int,
                               ϕmin::Real, ϕmax::Real, Nϕ::Int;
                               norm_tol::Real=1e-6)
    length(ψ_plus) == length(ψ_minus) ||
        throw(DimensionMismatch("ψ_plus and ψ_minus must have the same length: " *
                                "got $(length(ψ_plus)) and $(length(ψ_minus))"))
    iszero(δ) && throw(ArgumentError("central finite-difference step δ must be non-zero"))

    # Single diagonalization, shared by both finite-difference states.
    λx, V = eigv_xop(length(ψ_plus))

    P_plus  = _quadrature_distribution_from_basis(
        ψ_plus,  xmin, xmax, Nx, ϕmin, ϕmax, Nϕ, λx, V; norm_tol)
    P_minus = _quadrature_distribution_from_basis(
        ψ_minus, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ, λx, V; norm_tol)

    dP = (P_plus.P .- P_minus.P) ./ (2 * δ)

    return (xgrid = P_plus.xgrid, ϕgrid = P_minus.ϕgrid, dP = dP)
end