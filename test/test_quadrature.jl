# test/test_quadrature.jl
# Dedicated quick-check (fake it til I make it) for quadrature_distribution.
# Run directly:  julia --project test/test_quadrature.jl

using nonKam
using Test

@testset "Quadrature distribution" begin
    # Small N -> exact X eigensolver; keeps the test fast and cache-free.
    N, x0, p0 = 32, 1.0, 0.0

    # A Fock-state-ish input vector in a 32-mode basis.
    ψ = [zeros(ComplexF64, N - 1); [1.0]]

    res = quadrature_distribution(ψ, -4.0, 4.0, 129, -π, π, 16)

    x = res.xgrid
    ϕ = res.ϕgrid
    P = res.P

    @test x isa Vector{Float64}
    @test ϕ isa Vector{Float64}
    @test P isa Matrix{Float64}

    # Dimension checks: P is (Nx, Nϕ).
    @test length(x) == 129
    @test length(ϕ) == 16
    @test size(P) == (129, 16)

    # Each column is a valid probability vector: finite, non-negative, and
    # (since the quadrature_distribution no longer re-normalizes to unit area
    # on the sampled window) each column sums to at most 1 — exactly ‖ψ‖² = 1
    # only when the window [xmin, xmax] covers the full support.  The Fock
    # state |N−1⟩ with N = 32 has support well inside [−4, 4], so here columns
    # DO sum to 1 on the eigenbasis; the point is we no longer force a
    # per-column renormalization that would silently rescale away truncation.
    @test all(x -> all(isfinite, x), eachcol(P))
    @test all(col -> all(col .>= 0.0), eachcol(P))
    @test all(col -> sum(col) <= 1.0 + 1e-8, eachcol(P))

    # φ-grid covers the full [−π, π] window; x-grid is ordered.
    @test first(ϕ) ≈ -π
    @test last(ϕ) ≈ π
    @test issorted(x)

    # Sanity: rotating by ϕ = 2π reproduces the original state distribution.
    @test res.P[:, 1] ≈ res.P[:, end]
end

@testset "CFI never exceeds QFI" begin
    # Regression test for the CFI/QFI ratio crossing 1.
    #
    # The classical Fisher information of *any* measurement is bounded above by
    # the quantum Fisher information, CFI(θ) <= QFI(θ).  A ratio > 1 signals
    # noise/inconsistency — historically the per-column renormalization of
    # quadrature_distribution (unit-area on a *truncated* window, without an
    # identical rescale of the finite-difference derivative) inflated CFI.
    #
    # Use a displaced coherent state with a *deliberately truncated* x-window
    # so columns of P do NOT sum to 1 on the grid — exactly the regime where
    # the old code produced CFI > QFI.
    N, ω, x0, p0 = 64, 2 / π, -0.59, -8.14      # displaced, high-p state
    δ  = 1e-3                                    # FD step for CFI
    ϵ  = 1e-3                                    # FD step for QFI
    xw = 4.0                                     # x-window half-width

    ψ   = coherent_state(x0,     p0, ω, N)
    ψp  = coherent_state(x0 + δ, p0, ω, N)
    ψm  = coherent_state(x0 - δ, p0, ω, N)
    ψe  = coherent_state(x0 + ϵ, p0, ω, N)

    qfi = calculate_qfi_fidility(ψ, ψe, ϵ)

    res = quadrature_cfi(ψp, ψm, δ, -xw, xw, 201, -π, π, 17)

    # The window (-xw, xw) NARROWLY truncates the distribution of the displaced
    # state: columns of res.P should NOT sum to 1 on the grid.  This is what
    # makes the test bite — the bug lived in forcing that sum to 1.
    @test maximum([sum(col) for col in eachcol(res.P)]) < 1.0 - 1e-6

    # The CFI of the quadrature measurement must be a Fisher information, so
    # it must never beat the QFI.
    @test all(c -> c <= qfi + 1e-6, res.CFI)
    @test res.CFI_opt <= qfi + 1e-6
end