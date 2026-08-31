#!/usr/bin/env julia

"""
Sweep R across values very close to a base value, computing the mean
oscillator energy ⟨n̂⟩ vs time for a single number state, and fit the
intermediate power-law exponent m (t^m) for each R.

Run from the repository root:

    julia --project=. scripts/compute_mean_energy_R_sweep.jl

or load it interactively from the REPL (stays in the REPL afterwards):

    julia> include("scripts/compute_mean_energy_R_sweep.jl")

The R values are the base R plus, for each offset δ in
{10^-1, ..., 10^-7}, both R+δ and R-δ.  Together with R itself this gives
1 + 2*7 = 15 values.  The drive frequency ω is held fixed (default ω = 2/π)
for every R; changing R changes only the kicking period via
τ = 2π / (ω * R).

Optional environment variables:
    R=2 N=5000 T_MAX=500 N0=0 K=0.5 OMEGA=0.6366197723675814 \
        EXPONENT_MIN=-7 EXPONENT_MAX=-1 \
        TRANSIENT_FRACTION=0.0 MIN_ABS_SLOPE=0.05 SLOPE_TOL=0.05 \
        MODE=ask \
        julia --project=. scripts/compute_mean_energy_R_sweep.jl

Output:
    data/mean_energy_R_sweep.jld2   (set OUTPUT_FILE to override)

Persistence: the archive holds a `datasets` vector, one entry per distinct
parameter set (N, K, omega, n, t_max).  Each entry stores the per-R time
samples, mean-energy traces, fit exponent m, fitted curve, and fit window,
plus decoration parameters.

The existence check runs BEFORE any computation:

  * if no dataset matches the current (N, K, omega, n, t_max), the sweep runs
    and the new data is appended alongside existing data;
  * if a matching dataset already exists, the script asks whether to skip or
    overwrite.  Answer 's' (skip) to stop immediately without computing
    anything, or 'o' (overwrite) to recompute and replace that dataset.
    Skipping simply returns from the script, so the Julia REPL stays open
    when the script was loaded with include().

Setting the MODE environment variable to "skip" (or "s") / "overwrite"
(or "o") / "add" answers the question non-interactively.
"""

using nonKam
using Statistics
using JLD2

# Allow this script to be rerun from a REPL where `nonKam` was loaded before
# the piecewise-regime helper was added.  A fresh Julia session does not need
# this reload.
if !isdefined(nonKam, :find_power_law_regimes)
    Base.include(nonKam, joinpath(dirname(pathof(nonKam)), "find_converged_m.jl"))
end

function main()
    env_number(name, default, type) = parse(type, get(ENV, name, string(default)))

    # Kicked-harmonic-oscillator parameters.
    # ω is held fixed (default ω = 2/π); varying R changes only the kicking
    # period via τ = 2π / (ω * R), never the drive frequency.
    R_base = env_number("R", 2, Float64)
    ω = env_number("OMEGA", 2 / π, Float64)
    K = env_number("K", 0.5, Float64)

    # Simulation and initial-state parameters
    N = env_number("N", 1000, Int)
    t_max = env_number("T_MAX", 1000, Int)
    n = env_number("N0", 0, Int)

    # Fit-selection parameters (same meaning as in compute_mean_energy.jl)
    transient_fraction = env_number("TRANSIENT_FRACTION", 0.0, Float64)
    min_abs_slope = env_number("MIN_ABS_SLOPE", 0.05, Float64)
    slope_tol = env_number("SLOPE_TOL", 0.05, Float64)

    # Offsets applied to R: 10^p for p in [EXPONENT_MIN, ..., EXPONENT_MAX]
    exponent_min = env_number("EXPONENT_MIN", -7, Int)
    exponent_max = env_number("EXPONENT_MAX", -1, Int)

    N > 0 || throw(ArgumentError("N must be positive"))
    t_max > 0 || throw(ArgumentError("T_MAX must be positive"))
    0 <= n < N || throw(ArgumentError("N0 must satisfy 0 <= N0 < N"))
    exponent_min <= exponent_max || throw(ArgumentError("EXPONENT_MIN must be <= EXPONENT_MAX"))

    # R values: base, then base ± 10^p for each offset p.
    offsets = [10.0^p for p in exponent_min:exponent_max]
    R_values = sort(vcat([R_base], [R_base + δ for δ in offsets],
                         [R_base - δ for δ in offsets]))
    # Keep only positive R (a negative R is unphysical for this drive).
    R_values = filter(>(0.0), R_values)

    time = 0:(t_max - 1)

    # ----------------------------------------------------------------------
    # Persistence: decide BEFORE the heavy computation whether data already
    # exists for this parameter set, so a "skip" never runs the sweep.
    #
    # Structure of the JLD2 file:
    #   datasets :: Vector of NamedTuple, one entry per (N, K, omega, n, t_max)
    #   each entry has:
    #     signature :: (N, K, omega, n, t_max)   -- the identity of the dataset
    #     N, K, omega, n, t_max                  -- echoed for convenience
    #     runs       :: per-R NamedTuples (R, omega, tau, time, energy, m,
    #                                     fit_t, fit_energy)
    #     parameters :: decoration (R list, offsets, fit settings, ...)
    # ----------------------------------------------------------------------

    signature = (N=N, K=K, omega=ω, n=n, t_max=t_max)

    # Load any previously stored datasets (empty if the file does not exist yet).
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
    output_file = get(ENV, "OUTPUT_FILE", joinpath(output_dir, "mean_energy_R_sweep.jld2"))

    datasets = load_datasets(output_file)
    idx = find_dataset(datasets, signature)

    # ----------------------------------------------------------------------
    # Pre-computation check: if matching data already exists, ask before
    # running.  "skip" stops here, without computing anything, and simply
    # returns (so an interactive include() keeps the REPL open).
    # ----------------------------------------------------------------------
    if !isnothing(idx)
        decision = lowercase(get(ENV, "MODE", "ask"))
        if decision == "ask"
            println("Data already exists for (N, K, omega, n, t_max) = " *
                    "($N, $K, $ω, $n, $t_max) in ", normpath(output_file))
            print("Enter 's' to skip (keep existing data, stop) or " *
                  "'o' to overwrite (recompute and replace): ")
            decision = lowercase(strip(readline()))
            decision = decision in ("s", "skip") ? "skip" :
                       decision in ("o", "overwrite") ? "overwrite" : decision
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
        println("No existing data for (N, K, omega, n, t_max) = " *
                "($N, $K, $ω, $n, $t_max); running the sweep.")
    end

    # ----------------------------------------------------------------------
    # The sweep below is only reached when there is no matching dataset yet
    # or the user chose to overwrite.
    # ----------------------------------------------------------------------

    """
    Return the long, non-flat intermediate fit `a * t^m` for one trace.

    If no stable non-flat region is found (e.g. a drive for which the energy
    does not settle into a clean power law), return a fit with `m = NaN` and
    empty `t`/`energy` arrays instead of throwing, so a single "bad" R does
    not abort the whole sweep.
    """
    function intermediate_power_law(time, energy)
        try
            result = nonKam.find_converged_m(
                collect(time), energy;
                transient_fraction=transient_fraction,
                min_abs_slope=min_abs_slope,
                tol=slope_tol,
            )

            t_fit = result.t_converged
            fit_mask = (time .>= first(t_fit)) .& (time .<= last(t_fit)) .& (energy .> 0)
            t_plot = collect(float.(time[fit_mask]))
            energy_fit = energy[fit_mask]
            log_prefactor = mean(log.(energy_fit) .- result.m_converged .* log.(t_plot))
            return (m=result.m_converged, t=t_plot,
                    energy=exp(log_prefactor) .* t_plot .^ result.m_converged)
        catch err
            @warn "No power-law fit for this drive; marking m = NaN." exception=err
            return (m=NaN, t=Float64[], energy=Float64[])
        end
    end

    # Run the sweep.  ω is fixed for all R; only the kicking period
    # τ = 2π / (ω * R) changes with R.  A NamedTuple is stored per R so the
    # archive is self-describing.
    runs = map(R_values) do R
        τ = 2π / (ω * R)
        energy = mean_n(N, ω, τ, K, t_max, n)
        fit = intermediate_power_law(time, energy)
        m_str = isnan(fit.m) ? "NaN (no fit)" : string(round(fit.m, digits=6))
        t_str = isempty(fit.t) ? "no window" : string(first(fit.t), ":", last(fit.t))
        println("R = $(R): m = $m_str, t = $t_str")
        (R=R, omega=ω, tau=τ, time=collect(time), energy=energy,
         m=fit.m, fit_t=fit.t, fit_energy=fit.energy)
    end

    new_dataset = (signature=signature, N=N, K=K, omega=ω, n=n, t_max=t_max,
                   runs=runs,
                   parameters=(R=R_values, R_base=R_base, offsets=offsets,
                               fit_settings=(transient_fraction=transient_fraction,
                                             min_abs_slope=min_abs_slope,
                                             slope_tol=slope_tol)))

    # Store: append the new dataset, or replace the matching one in place.
    if isnothing(idx)
        push!(datasets, new_dataset)
        action = "added"
    else
        datasets[idx] = new_dataset
        action = "overwrote"
    end

    jldsave(output_file; datasets=datasets)
    println("$action dataset for (N, K, omega, n, t_max) = ($N, $K, $ω, $n, $t_max) in ", normpath(output_file))
    println("Total datasets stored: $(length(datasets))")
end

main()