# src/measurement/quadrature_distribution.jl
"""
    quadrature_distribution(ψ, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ)

Compute the position-quadrature probability distribution of the input state
`ψ` — and its family of quadrature-rotated copies — for use in computing the
(classical) Fisher information.

For each rotation phase ``ϕ`` the state is rotated by the number-operator
phase gate ``e^{-iϕ n̂}`` (cheap because ``n̂ = diag(0, 1, …, N-1)`` is
diagonal), then projected onto the position-quadrature eigenstates ``|x_i⟩``
of ``X = (a + a†)/√2``.  For each ``ϕ`` the resulting distribution is

    P(x_i | ϕ) = |⟨x_i| e^{-iϕ n̂} ψ⟩|².

# Arguments
- `ψ::AbstractVector{<:Complex}`: state in the Fock/number basis, length `N`.
- `xmin, xmax, Nx`: the position grid — `x ∈ range(xmin, stop=xmax, length=Nx)`.
- `ϕmin, ϕmax, Nϕ`: the rotation-angle grid — `ϕ ∈ range(ϕmin, stop=ϕmax, length=Nϕ)`.
- `norm_tol::Real=1e-6`: tolerance for the normalization warning (see below).

The `e^{-iϕ n̂}` rotation is implemented elementwise (one broadcast over the
diagonal ``n̂``), so the full family of rotated states is built in a single
pass instead of one loop per ϕ.

For each ``ϕ`` the code checks the **exact discrete** sum
``Σ_i |⟨x_i|ψ_ϕ⟩|² == ‖ψ‖²`` (equal to 1 for a normalized ``ψ``); if it deviates
from `norm_tol`, a warning is emitted with the offending `ϕ`.

The returned `P` is **not** renormalized to unit area on the sampled window:
columns keep the exact eigenbasis normalization ``Σ_i P[i,j] = ‖ψ‖²``.  With a
normalized input, a column sums to 1 **only if** the `x`-window
`[xmin, xmax]` covers the full support of ``P(x|ϕ)``; when the window
truncates the distribution the column sum is `< 1`.  This per-column
normalization is deliberately *not* forced back to 1: CFI is built from
`Σ (∂P/∂θ)² / P` using the *same* normalization for `P` and `∂P/∂θ`, and an
artificial per-ϕ rescaling of `P` alone would inflate CFI relative to QFI
(CFI(θ) ≤ QFI(θ) for any measurement — a ratio > 1 signals noise, not physics).

# Returns
A `NamedTuple`
```
<xgrid = x         ::Vector{Float64},   # position grid
 ϕgrid = ϕ         ::Vector{Float64},   # rotation-angle grid
 P     = P         ::Matrix{Float64})   # P[x, ϕ], size (Nx, Nϕ)
```
Each column `P[:, j]` is the position distribution for `ϕ = ϕgrid[j]`,
normalized to ``‖ψ‖²`` on the exact X eigenbasis (unit area only when the
window is wide enough to cover the full support).
"""
function quadrature_distribution(ψ::AbstractVector{<:Complex},
                                 xmin::Real, xmax::Real, Nx::Int,
                                 ϕmin::Real, ϕmax::Real, Nϕ::Int;
                                 norm_tol::Real=1e-6)
    # Position-quadrature eigenbasis (cached internally by eigv_xop for large N).
    λx, V = eigv_xop(length(ψ))

    return _quadrature_distribution_from_basis(
        ψ, xmin, xmax, Nx, ϕmin, ϕmax, Nϕ, λx, V; norm_tol)
end

# Internal core: compute the distribution for a *given* X eigenbasis (λx, V).
# Kept as a separate entry point so the eigendecomposition (the expensive
# O(N³) step, cached on disk for large N) is performed once and then shared
# between the two states of a central finite difference.
function _quadrature_distribution_from_basis(ψ::AbstractVector{<:Complex},
                                             xmin::Real, xmax::Real, Nx::Int,
                                             ϕmin::Real, ϕmax::Real, Nϕ::Int,
                                             λx::Vector{Float64}, V;
                                             norm_tol::Real=1e-6)
    N = length(ψ)

    # ---- 2. Rotation-angle grid and phase-gate vectors ----
    ϕgrid   = range(ϕmin, stop=ϕmax, length=Nϕ) |> collect
    n_diag  = 0:(N-1)            # number operator is diagonal: n̂ = diag(0,1,…,N-1)

    # ---- 3. Build all rotated states in one pass: ψ_ϕ[j] = e^{-iϕ_j n̂} ψ ----
    # n̂ is diagonal, so each rotated state is the elementwise phase gate
    #   (ψ_ϕ[j])[n] = e^{-i ϕ_j n} ψ[n],
    # one row per ϕ, one column per Fock index n.
    ψ_row = transpose(ψ)                       # shape (1, N), matches n_diag
    psi_phis = exp.(-im .* ϕgrid .* n_diag') .* ψ_row   # size (Nϕ, N)
    # For each ϕ_j: psi_phis[j, :] is the Fock-coefficient vector of e^{-iϕ_j n̂} ψ.

    # ---- 4. Project onto X eigenstates and take |·|²  ----
    # ⟨x_i|ψ_ϕ⟩  = (V' * ψ_ϕ)[i]
    Pmatrix = real.(abs2.(V' * psi_phis'))         # size (N, Nϕ): P[x_i, ϕ_j]

    # ---- 4b. Sanity check: discrete probability sum must equal ||ψ||² (§1.) ----
    # For each ϕ, Σ_i |⟨x_i|ψ_ϕ⟩|² = ⟨ψ|ψ⟩, which is 1 only for a normalized
    # input. This is the physically honest check — done on the exact eigenbasis,
    # before any resampling, so it can actually catch a bad input.
    norm2 = sum(abs2, ψ)                           # ||ψ||²
    for j in 1:Nϕ
        s = sum(@view Pmatrix[:, j])
        if abs(s - norm2) > norm_tol
            @warn "Quadrature distribution not normalized" ϕ = ϕgrid[j] s norm2
        end
    end

    # ---- 5. Interpolate / re-sample onto the requested  x-grid  ----
    # The discrete eigenvalues λx are exact X values; we interpolate them onto
    # the dense grid x∈[xmin,xmax] requested by the caller.
    xn       = range(xmin, stop=xmax, length=Nx) |> collect
    Pgrid    = _interp_onto_grid(λx, Pmatrix, xn)

    # ---- 6. Do NOT renormalize columns here. ----
    # The physical probability mass was already fixed exactly by the X-eigenbasis
    # projection in step 4 (each column sums to ||ψ||²; see the step-4b check).
    # Renormalizing each column to unit area on the *sampled* window would scale
    # every rotated column by an arbitrary, ϕ-dependent factor s_j = Σ_i Pgrid[i,j].
    #
    # CFI is built as Σ (∂P/∂θ)² / P with both the derivative and P taken from
    # the same eigenbasis normalization.  Renormalizing P alone — while ∂P/∂θ is
    # the finite difference of the *un*renormalized columns — is inconsistent and
    # inflates CFI relative to QFI, which is exactly how the ratio CFI/QFI could
    # spuriously cross 1 for a quadrature measurement (CFI(θ) ≤ QFI(θ) always).
    #
    # The returned P therefore may NOT sum to 1 over [xmin, xmax]; that is
    # expected when the window truncates the distribution.

    return (xgrid = xn, ϕgrid = ϕgrid, P = Pgrid)
end

"""
    _interp_onto_grid(λx, Pmatrix, xn)

Linearly interpolate the per-column distributions (defined on the discrete X
eigenvalues `λx`) onto the uniform position grid `xn`.  Values outside the
eigenvalue range are clamped to the boundary columns via flat/nearest fill.
This produces the `(Nx, Nϕ)` matrix `Pgrid` without building any dense X
operator.
"""
function _interp_onto_grid(λx::Vector{Float64}, Pmatrix::Matrix{Float64},
                           xn::Vector{Float64})
    Nx  = length(xn)
    Nϕ  = size(Pmatrix, 2)
    Pgrid = Array{Float64}(undef, Nx, Nϕ)

    xlo, xhi = extrema(λx)

    @inbounds for j in 1:Nϕ
        col = @view Pmatrix[:, j]
        for i in 1:Nx
            x = xn[i]
            # Clamp to the first/last eigenvalue index.
            if x <= xlo
                Pgrid[i, j] = col[1]
            elseif x >= xhi
                Pgrid[i, j] = col[end]
            else
                # Locate the bracket [k, k+1] of eigenvalues around x.
                # searchsortedfirst gives the first index with λx[idx] >= x,
                # so the neighbouring grid points are k-1 and k.
                k = searchsortedfirst(λx, x)
                k = clamp(k, 2, length(λx))     # ensure k-1 is a valid lambda index
                lo, hi = k - 1, k
                # Fractional position between the two fixed eigenvalues.
                frac = (x - λx[lo]) / (λx[hi] - λx[lo])
                Pgrid[i, j] = (1 - frac) * col[lo] + frac * col[hi]
            end
        end
    end

    return Pgrid
end