#!/usr/bin/env julia

"""
Compute the QFI time traces for R values very close to a base R0, for two
initial states (a Fock/number state and a coherent state) under one shared
parameter set, and store everything (append / partial-overwrite) in one
self-describing JLD2 archive.

Run from the repository root:

    julia --project=. scripts/qfi_time_R_sweep.jl

or load it interactively from the REPL (stays in the REPL afterwards):

    julia> include("scripts/qfi_time_R_sweep.jl")

The R values are chosen in one of two ways (toggle inside the script with #):
  * the mirror sweep: R0 plus, for each offset δ in {10^-1, ..., 10^-7},
    both R0+δ and R0-δ — 1 + 2*7 = 15 values;
  * a linear grid from R_MIN to R_MAX (symmetric about R0) with N_R values,
    so the integer R0 is always the midpoint of the grid.
The drive frequency ω is held fixed (default ω = 2/π) for every R; changing R
changes only the kicking period via τ = 2π / (ω * R).

Optional environment variables:
    R0=2 OMEGA=0.6366197723675814 K=0.5 EPSILON=1e-6 N=1000 T_MAX=1000 \
        X0=-0.5903 P0=-8.1437 EXPONENT_MIN=-7 EXPONENT_MAX=-1 \
        COHERENT=true MODE=ask \
        julia --project=. scripts/qfi_time_R_sweep.jl

Output:
    data/qfi_time_R_sweep.jld2   (set OUTPUT_FILE to override)

Archive layout: a top-level `datasets` vector, one NamedTuple per distinct
parameter set (N, K, omega, epsilon, t_max).  Each entry stores the initial
states as qfi["Fock"] and qfi["Coherent"] (a Dict keyed by state name);
each value is a NamedTuple (R, tau, time_steps, times, qfi) in QFI sweep
format, qfi[time_step, R_index].

The existence check runs BEFORE any computation:

  * no matching dataset -> the sweep runs and the new entry is appended
    alongside any existing data;
  * a matching dataset exists -> ask 's' (skip / keep existing, stop) or
    'o' (overwrite / recompute and replace that entry only).  Skipping
    simply returns so an interactive include() keeps the REPL open.

MODE=skip|overwrite answers the question non-interactively.
"""

using nonKam
using JLD2

# Allow this script to be rerun from a REPL where `nonKam` was loaded before
# the piecewise-regime helper was added.  A fresh Julia session does not need
# this reload.
if !isdefined(nonKam, :find_power_law_regimes)
    Base.include(nonKam, joinpath(dirname(pathof(nonKam)), "find_converged_m.jl"))
end

function main()
    env_number(name, default, type) = parse(type, get(ENV, name, string(default)))
    env_bool(name, default) = lowercase(get(ENV, name, string(default))) in
                              ("true", "1", "yes", "y")

    # Kicked-harmonic-oscillator parameters.
    # ω is held fixed (default ω = 2/π); varying R changes only the kicking
    # period via τ = 2π / (ω * R), never the drive frequency.
    R0 = env_number("R0", 2, Float64)
    ω = env_number("OMEGA", 2 / π, Float64)
    K = env_number("K", 0.5, Float64)
    ϵ = env_number("EPSILON", 1e-6, Float64)

    # Simulation and initial-state parameters
    N = env_number("N", 26000, Int)
    t_max = env_number("T_MAX", 200, Int)
    x0 = env_number("X0", 0.0121+3*π, Float64)
    p0 = env_number("P0", 0.0121, Float64)
    n0 = env_number("N0", 0, Int)

    # Offsets applied to R0: 10^p for p in [EXPONENT_MIN, ..., EXPONENT_MAX]
    exponent_min = env_number("EXPONENT_MIN", -7, Int)
    exponent_max = env_number("EXPONENT_MAX", -1, Int)

    N > 0 || throw(ArgumentError("N must be positive"))
    t_max > 0 || throw(ArgumentError("T_MAX must be positive"))
    0 <= n0 < N || throw(ArgumentError("N0 must satisfy 0 <= N0 < N"))
    exponent_min <= exponent_max || throw(ArgumentError("EXPONENT_MIN must be <= EXPONENT_MAX"))

    # R values: two choices, toggle by commenting/uncommenting with #.
    #   * Choice 1 (default): R0, then R0 ± 10^p for each offset p (mirrors the
    #     mean-energy sweep so both sweeps probe the same resonance neighbourhood).
    #   * Choice 2: a linear grid from R_MIN to R_MAX with N_R values.  R_MIN
    #     and R_MAX must be symmetric about R0 (e.g. R0 - a and R0 + a) so that
    #     the integer R0 is exactly the midpoint of the grid.  This choice
    #     ignores EXPONENT_MIN / EXPONENT_MAX.
    # ------------------------------------------------------------------------
    # `offsets` is only meaningful for Choice 1, but it is also stored in the
    # archive's `parameters` field below, so define it whether or not Choice 1
    # is the active grid.
    offsets = [10.0^p for p in exponent_min:exponent_max]

    # Choice 1 (default): mirror sweep around R0.  Uncomment to activate.
    # R = sort(vcat([R0], [R0 + δ for δ in offsets], [R0 - δ for δ in offsets]))

    # Choice 2: linear grid centred on R0 with N_R values (odd, so R0 is the
    # exact middle) spanning [R0 - A, R0 + A].
    N_R = 41    # number of R values (odd, so R0 lands in the middle)
    A   = 0.001   # half-width; grid covers [R0 - A, R0 + A]
    R = collect(range(R0 - A, R0 + A; length=N_R))
    # ------------------------------------------------------------------------

    R = filter(>(0.0), R)  # keep only positive R

    # ----------------------------------------------------------------------
    # Persistence: decide BEFORE the heavy computation whether data already
    # exists for this parameter set, so a "skip" never runs the sweep.
    #
    # Archive structure (top level):
    #   datasets :: Vector of NamedTuple, one entry per (N, K, omega,
    #               epsilon, t_max) parameter set:
    #     signature :: that 5-tuple                   -- identity
    #     N, K, omega, epsilon, t_max                 -- echoed for convenience
    #     states   :: Symbol list of computed states  -- e.g. (:Fock, :Coherent)
    #     qfi      :: Dict{String, NamedTuple} keyed "Fock"/"Coherent", each
    #                 value in floquet_qfi_R_sweep form: (R, tau, time_steps,
    #                 times, qfi) with qfi[time_step, R_index]
    #     parameters :: decoration (R grid, offsets, ...)
    # ----------------------------------------------------------------------

    signature = (N=N, K=K, omega=ω, epsilon=ϵ, t_max=t_max)

    # Load any previously stored datasets (empty if the file does not exist).
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
    output_file = get(ENV, "OUTPUT_FILE", joinpath(output_dir, "qfi_time_R_sweep.jld2"))

    datasets = load_datasets(output_file)
    idx = find_dataset(datasets, signature)

    # ----------------------------------------------------------------------
    # Pre-computation check: if matching data already exists, ask before
    # running.  "skip" stops here without computing anything and simply
    # returns (an interactive include() keeps the REPL open).
    # ----------------------------------------------------------------------
    if !isnothing(idx)
        decision = lowercase(get(ENV, "MODE", "ask"))
        if decision == "ask"
            println("Data already exists for (N, K, omega, eps, t_max) = " *
                    "($N, $K, $ω, $ϵ, $t_max) in ", normpath(output_file))
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
        println("No existing data for (N, K, omega, eps, t_max) = " *
                "($N, $K, $ω, $ϵ, $t_max); running the sweep.")
    end

    # ----------------------------------------------------------------------
    # The computation below is only reached when there is no matching dataset
    # yet or the user chose to overwrite.
    # ----------------------------------------------------------------------

    # Fock / number state |n0>
    fock_sweep = floquet_qfi_R_sweep(N, ω, R, K, ϵ, t_max, Int(n0);
                                      progress=true)

    # Coherent state |x0, p0> (only run when requested; default on so a fresh
    # run computes both states and fills the single-signature archive).
    coherent_on = env_bool("COHERENT", true)
    if coherent_on
        coherent_sweep = floquet_qfi_R_sweep(N, ω, R, K, ϵ, t_max, (x0, p0);
                                              progress=true)
    end

    qfi = Dict{String,Any}()
    qfi["Fock"] = fock_sweep
    if coherent_on
        qfi["Coherent"] = coherent_sweep
    end

    new_dataset = (signature=signature, N=N, K=K, omega=ω, epsilon=ϵ,
                   t_max=t_max,
                   states=sort!(collect(keys(qfi))),
                   qfi=qfi,
                   parameters=(R=R, R0=R0, offsets=offsets,
                               coherent_state=(x0=x0, p0=p0, n0=n0)))

    # Store: append the new dataset, or replace the matching one in place.
    if isnothing(idx)
        push!(datasets, new_dataset)
        action = "added"
    else
        datasets[idx] = new_dataset
        action = "overwrote"
    end

    jldsave(output_file; datasets=datasets)
    println("$action dataset for (N, K, omega, eps, t_max) = " *
            "($N, $K, $ω, $ϵ, $t_max) in ", normpath(output_file))
    println("Total datasets stored: $(length(datasets))  " *
            "($(join(new_dataset.states, ", ")))")
end

main()