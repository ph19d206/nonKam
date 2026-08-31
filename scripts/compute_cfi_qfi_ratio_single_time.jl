#!/usr/bin/env julia

"""
Compute the derived ratio CFI / QFI for a single, fixed integer time step t,
starting from a **coherent state**, and store the result in a self-describing
JLD2 archive.

  - CFI  : classical Fisher information of the position-quadrature measurement
           at rotation angle ϕ, estimated by central finite difference of the
           initial state with respect to the parameter θ = ω (the drive
           frequency).  Computed with `quadrature_cfi`, which returns a
           per-ϕ vector.
  - QFI  : quantum Fisher information for the same θ = ω, computed from the
           fidelity susceptibility via `calculate_qfi_fidility`.
  - ratio: CFI[ϕ] / QFI, kept **per ϕ value** (not maximised) so it can be
           plotted against ϕ later.

For each time step t we evolve the coherent state |ψ0(x0,p0)⟩ under the
kicked-harmonic-oscillator flush with the `qfi_time_R_sweep` convention:
U_free = exp(-i ω τ n̂) at frequency ω, then the instantaneous kick from the
diagonalised `eigv_kikU(N, ω, K)` basis.  The time step is an integer number
of kicks, fixed by T_STEP.

Run from the repository root:

    julia --project=. scripts/compute_cfi_qfi_ratio_single_time.jl

or load interactively (stays in the REPL after completion):

    julia> include("scripts/compute_cfi_qfi_ratio_single_time.jl")

Optional environment variables (defaults shown):

    N=400          Number of Fock levels
    OMEGA=2/pi     Drive frequency ω (the estimated parameter θ)
    K=0.5          Kick strength
    EPSILON=1e-2   Perturbation of ω used for the QFI fidelity susceptibility
    DELTA=1e-3     Finite-difference step of ω used for the central-difference CFI
    R=2            Floquet parameter; τ = 2π/(ω*R)
    X0=-0.5903     Coherent-state position centre
    P0=-8.1437     Coherent-state momentum centre
    T_STEP=10      Fixed integer number of kicks / time steps to evolve
    XMIN=-4        Position grid lower bound for the quadrature measurement
    XMAX= 4        Position grid upper bound
    NX=193         Number of position grid points
    PHI_MIN=-pi    Rotation-angle grid lower bound
    PHI_MAX= pi    Rotation-angle grid upper bound
    NPHI=101       Number of rotation angles
    ZERO_THRESH=0  Quadrature-CFI saddle over bins with P < ZERO_THRESH
    OUTPUT_FILE=data/cfi_qfi_ratio_single_time.jld2

Output:

    data/cfi_qfi_ratio_single_time.jld2

Archive layout: a top-level `datasets` vector, one NamedTuple per distinct
parameter set (N, K, omega, epsilon, R, x0, p0).  Each entry stores:

    signature         :: the identity tuple above
    N, K, omega, epsilon, R, x0, p0, tau
    time_steps        :: [T_STEP]
    times             :: [T_STEP]
    phi               :: vector of rotation angles (length NPHI)
    cfi               :: matrix cfi[phi_index, time_index]
    qfi               :: vector qfi[time_index]
    ratio             :: matrix ratio[phi_index, time_index] = cfi ./ qfi

The existence check runs BEFORE any computation:

  * no matching dataset -> the sweep runs and the new entry is appended
    alongside any existing data;
  * a matching dataset exists -> ask 's' (skip / keep existing, stop) or
    'o' (overwrite / recompute and replace that entry only).  Skipping simply
    returns so an interactive include() keeps the REPL open.

MODE=skip|overwrite answers the question non-interactively.
"""

using nonKam
using JLD2
using Dates
using LinearAlgebra

function main()
    env_number(name, default, type) =
        haskey(ENV, name) ? parse(type, ENV[name]) : convert(type, default)

    # Kicked-harmonic-oscillator parameters.
    N = env_number("N", 5000, Int)
    ω = env_number("OMEGA", 2 / π, Float64)
    K = env_number("K", 0.5, Float64)
    ϵ = env_number("EPSILON", 1e-2, Float64)
    R = env_number("R", 2, Float64)
    τ = 2π / (ω * R)
    x0 = env_number("X0", -0.5903, Float64)
    p0 = env_number("P0", -8.1437, Float64)
    t_step = env_number("T_STEP", 10, Int)

    # CFI finite-difference step in the parameter θ = ω.
    δ = env_number("DELTA", 1e-2, Float64)

    # Quadrature-measurement grid.
    xmin = env_number("XMIN", -4, Float64)
    xmax = env_number("XMAX", 4, Float64)
    nx    = env_number("NX", 193, Int)
    ϕmin  = env_number("PHI_MIN", -π, Float64)
    ϕmax  = env_number("PHI_MAX", π, Float64)
    nϕ    = env_number("NPHI", 201, Int)
    zero_thresh = env_number("ZERO_THRESH", 0, Float64)

    N > 0 || throw(ArgumentError("N must be positive"))
    t_step > 0 || throw(ArgumentError("T_STEP must be positive"))
    R > 0 || throw(ArgumentError("R must be positive"))
    0 < ϵ || throw(ArgumentError("EPSILON must be positive"))
    0 < δ || throw(ArgumentError("DELTA must be positive"))

    # Identity of this dataset. Only K, R, omega, N, x0, p0 participate in the
    # skip/overwrite check; EPSILON is intentionally excluded because it is a
    # numerical-analysis knob (the CFI finite-difference perturbation) rather
    # than part of the physical parameter set.
    signature = (N=N, K=K, omega=ω, R=R, x0=x0, p0=p0)

    # ----------------------------------------------------------------------
    # Persistence: decide BEFORE the heavy computation whether data already
    # exists for this parameter set, so a "skip" never runs the evolution.
    # ----------------------------------------------------------------------
    function load_datasets(file)
        isfile(file) || return Any[]
        try
            jldopen(file, "r") do f
                haskey(f, "datasets") ? Vector(f["datasets"]) : Any[]
            end
        catch
            return Any[]  # unreadable/partial file -> start fresh
        end
    end

    function find_dataset(datasets, sig)
        for (i, ds) in enumerate(datasets)
            ds.signature == sig && return i
        end
        return nothing
    end

    output_dir = joinpath(@__DIR__, "..", "data")
    mkpath(output_dir)
    output_file = get(ENV, "OUTPUT_FILE",
                      joinpath(output_dir, "cfi_qfi_ratio_single_time.jld2"))

    datasets = load_datasets(output_file)
    idx = find_dataset(datasets, signature)

    if !isnothing(idx)
        decision = lowercase(get(ENV, "MODE", "ask"))
        if decision == "ask"
            println("Data already exists for (N, K, omega, eps, R, x0, p0) = " *
                    "($N, $K, $ω, $ϵ, $R, $x0, $p0) at t = $t_step in ",
                    normpath(output_file))
            print("Enter 's' to skip (keep existing data, stop) or " *
                  "'o' to overwrite (recompute and replace): ")
            decision = lowercase(strip(readline()))
        end
        decision = decision in ("s", "skip") ? "skip" :
                   decision in ("o", "overwrite") ? "overwrite" : decision
        if decision == "overwrite"
            println("Overwriting the existing dataset.")
        elseif decision in ("skip", "add")
            println("Skipping: existing data kept. Nothing was computed or written.")
            return
        else
            error("Unknown decision '$(decision)'. Use 's'/'skip' or 'o'/'overwrite'.")
        end
    else
        println("No existing data for (N, K, omega, eps, R, x0, p0) = " *
                "($N, $K, $ω, $ϵ, $R, $x0, $p0) at t = $t_step; running.")
    end

    # ----------------------------------------------------------------------
    # Evolution machinery under the kicked-HO convention.
    #   U_free = exp(-i ω τ n̂) ; kick via eigv_kikU(N, ω, K) -> (λ, V).
    #
    # Parameter dependence. The estimated parameter is θ = ω.  Following the
    # `floquet_qfi` perturbation convention, a small change in ω enters the
    # dynamics ONLY through the free-evolution phase: U_free(N, ω, τ) =
    # exp(-i ω τ n̂) ; the kick operator (D_kick, V) is held fixed (built from
    # the nominal ω, NOT rebuilt at the perturbed frequency).
    #
    #   QFI(θ) : needs ψ(t) and its perturbed partner ψ_ϵ(t), using
    #            U_free(N, ω + ϵ, τ) (per the engine);   fidelity susceptibility.
    #   CFI(θ) : central difference  (ψ(t; ω+δ) - ψ(t; ω-δ)) / (2δ),  using
    #            U_free(N, ω ± δ, τ);   from `quadrature_cfi`, per rotation ϕ.
    # ----------------------------------------------------------------------
    λ, V       = eigv_kikU(N, ω, K)
    D_kick     = Diagonal(λ)
    U_free     = free_U(N, ω,     τ)
    U_free_ϵ   = free_U(N, ω + ϵ, τ)
    U_free_cfi_p = free_U(N, ω + δ, τ)
    U_free_cfi_m = free_U(N, ω - δ, τ)

    ψ0 = coherent_state(x0, p0, ω, N)

    # ---- buffers ---------------------------------------------------------
    work() = [zeros(ComplexF64, N), zeros(ComplexF64, N)]  # [buf1, buf2] per state

    # QFI paths (unperturbed + perturbed).
    ψq  = copy(ψ0);  qbuf  = work()
    ψe  = copy(ψ0);  ebuf  = work()

    # CFI paths (ω+δ and ω-δ central-difference partners).
    ψp  = copy(ψ0);  pbuf  = work()
    ψm  = copy(ψ0);  mbuf  = work()

    # ---- time evolution, t = 1 .. t_step --------------------------------
    for t in 1:t_step
        evolve!(U_free,      D_kick, V, ψq, qbuf[1], qbuf[2])
        evolve!(U_free_ϵ,    D_kick, V, ψe, ebuf[1], ebuf[2])
        evolve!(U_free_cfi_p, D_kick, V, ψp, pbuf[1], pbuf[2])
        evolve!(U_free_cfi_m, D_kick, V, ψm, mbuf[1], mbuf[2])
    end

    # ---- Quantum Fisher information (θ = ω) at the final time step -------
    qfi = calculate_qfi_fidility(ψq, ψe, ϵ)

    # ---- Classical Fisher information (θ = ω), per rotation angle ϕ ------
    cfi = quadrature_cfi(ψp, ψm, δ, xmin, xmax, nx, ϕmin, ϕmax, nϕ;
                         zero_thresh = zero_thresh)

    # ---- Derived ratio, kept per-ϕ (not maximised) ----------------------
    iszero(qfi) && @warn "QFI is zero; ratio undefined/diverges" qfi N K ω τ t_step
    ratio = cfi.CFI ./ qfi                     # length nϕ

    # ----------------------------------------------------------------------
    # Persist: append new dataset, or replace in place when overwriting.
    # ----------------------------------------------------------------------
    dataset = (
        signature = signature,
        N = N, K = K, omega = ω, epsilon = ϵ, R = R, x0 = x0, p0 = p0,
        tau = τ, delta = δ,
        time_steps = [t_step],
        times      = [t_step],
        phi        = cfi.ϕgrid,
        cfi        = reshape(cfi.CFI, :, 1),     # [nϕ, 1]
        qfi        = [qfi],                      # length-1 vector
        ratio      = reshape(ratio, :, 1),       # [nϕ, 1]
    )

    if idx === nothing
        push!(datasets, dataset)
        println("Appended new dataset for (N, K, omega, eps, R, x0, p0, t) = " *
                "($N, $K, $ω, $ϵ, $R, $x0, $p0, $t_step).")
    else
        datasets[idx] = dataset
        println("Replaced dataset for (N, K, omega, eps, R, x0, p0, t) = " *
                "($N, $K, $ω, $ϵ, $R, $x0, $p0, $t_step).")
    end

    jldopen(output_file, "w") do f
        f["datasets"] = datasets
        f["created_at"] = string(Dates.now())
    end
    println("Saved to ", normpath(output_file))
end

main()