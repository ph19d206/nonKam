#!/usr/bin/env julia

"""
Compute the mean oscillator energy ⟨n̂⟩ and the intermediate power-law exponent
m, then save everything to a self-describing JLD2 archive in the data folder.

Run from the repository root:

    julia --project=. scripts/compute_mean_energy.jl

or load it interactively from the REPL (stays in the REPL afterwards):

    julia> include("scripts/compute_mean_energy.jl")

Optional environment variables:
    N=5000 T_MAX=500 N0=0 K=0.5 R=2 X0=-0.5903 P0=-8.1437 \
        SCALE=loglog TRANSIENT_FRACTION=0.0 MIN_ABS_SLOPE=0.05 SLOPE_TOL=0.05 \
        julia --project=. scripts/compute_mean_energy.jl

Output:
    data/mean_energy_vs_time.jld2   (set OUTPUT_FILE to override)
All runs live in one archive as a `datasets` vector; an entry is added per
distinct parameter set (N, R, omega, x0, p0, n). Reruns with different
parameters append a new entry without erasing older data. If a matching
entry already exists, you are prompted to overwrite or skip before the
simulation runs. Skipping stops immediately and stays in the REPL (it never
runs the simulation, so old data is preserved).
The stored samples correspond to t = 0, ..., t_max - 1, which are also the
points where ⟨n̂⟩ data is defined.

`TRANSIENT_FRACTION`, `MIN_ABS_SLOPE`, and `SLOPE_TOL` tune selection of the
single intermediate scaling region. Only that long, non-flat region is used
for the fit; the short-time transient and final finite-basis saturation are
excluded.

The plotting script scripts/plot_mean_energy_vs_time.jl loads the archive this
script produces.
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

    # -------------Input parameters---------------
    # Kicked-harmonic-oscillator parameters.
    # ω is held fixed (default ω = 2/π); R sets the kicking period via
    # τ = 2π / (ω * R).  Changing R changes only τ, never ω.
    R = env_number("R", 4, Int)
    ω = env_number("OMEGA", 2 / π, Float64)
    τ = 2π / (ω * R)
    K = env_number("K", 1.5, Float64)

    # Simulation and initial-state parameters
    N = env_number("N", 25000, Int)
    t_max = env_number("T_MAX", 1000, Int)
    n = env_number("N0", 0, Int)
    x0 = env_number("X0", -0.5903, Float64)
    p0 = env_number("P0", -8.1437, Float64)
    time = 0:(t_max - 1)
    scale = Symbol(get(ENV, "SCALE", "loglog"))
    transient_fraction = env_number("TRANSIENT_FRACTION", 0.0, Float64)
    min_abs_slope = env_number("MIN_ABS_SLOPE", 0.05, Float64)
    slope_tol = env_number("SLOPE_TOL", 0.05, Float64)
    # ----------------------------

    N > 0 || throw(ArgumentError("N must be positive"))
    t_max > 0 || throw(ArgumentError("T_MAX must be positive"))
    0 <= n < N || throw(ArgumentError("N0 must satisfy 0 <= N0 < N"))

    output_dir = joinpath(@__DIR__, "..", "data")
    mkpath(output_dir)
    output_file = get(ENV, "OUTPUT_FILE", joinpath(output_dir, "mean_energy_vs_time.jld2"))

    # All runs live in one flat archive as a `datasets` vector, one entry per
    # distinct parameter set.  Existence is checked before the simulation so a
    # "skip" wastes no compute.
    signature = (N=N, R=R, omega=ω, x0=x0, p0=p0, n=n)

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

    datasets = load_datasets(output_file)
    idx = find_dataset(datasets, signature)

    if !isnothing(idx)
        decision = lowercase(get(ENV, "MODE", "ask"))
        if decision == "ask"
            println("Data already exists for parameters (N, R, omega, x0, p0, n) = " *
                    "($N, $R, $ω, $x0, $p0, $n) in ", normpath(output_file))
            print("Enter 's' to skip (keep existing data, stop) or " *
                  "'o' to overwrite (recompute and replace): ")
            decision = lowercase(strip(readline()))
            decision = decision in ("s", "skip") ? "skip" :
                       decision in ("o", "overwrite") ? "overwrite" : decision
        end
        if decision == "overwrite"
            println("Overwriting the existing dataset.")
        else
            println("Skipping: existing data kept. Nothing was computed or written.")
            return  # stop here; stay in the REPL when loaded with include()
        end
    else
        println("No existing data for parameters (N, R, omega, x0, p0, n) = " *
                "($N, $R, $ω, $x0, $p0, $n); running the simulation.")
    end

    # ⟨n̂⟩ is the energy in oscillator-number units. Multiply by ℏω for
    # physical energy units if required.
    energy_number = mean_n(N, ω, τ, K, t_max, n)
    energy_coherent = mean_n(N, ω, τ, K, t_max, (x0, p0))

    """Return the long, non-flat intermediate fit `a * t^m` for one trace."""
    function intermediate_power_law(time, energy)
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
    end

    number_fit = intermediate_power_law(time, energy_number)
    coherent_fit = intermediate_power_law(time, energy_coherent)

    new_entry = (signature=signature, N=N, t_max=t_max, n=n, x0=x0, p0=p0, R=R,
                 omega=ω, tau=τ, K=K, scale=scale,
                 time=collect(time),
                 energy_number=energy_number,
                 energy_coherent=energy_coherent,
                 number_fit=number_fit,
                 coherent_fit=coherent_fit,
                 fit_settings=(transient_fraction=transient_fraction,
                               min_abs_slope=min_abs_slope, slope_tol=slope_tol))

    if isnothing(idx)
        push!(datasets, new_entry)
        action = "added"
    else
        datasets[idx] = new_entry
        action = "overwrote"
    end

    jldsave(output_file; datasets=datasets)

    println("$action dataset for parameters (N, R, omega, x0, p0, n) = " *
            "($N, $R, $ω, $x0, $p0, $n) in ", normpath(output_file))
    println("Number-state intermediate slope: m = $(number_fit.m), t = $(first(number_fit.t)):$(last(number_fit.t))")
    println("Coherent-state intermediate slope: m = $(coherent_fit.m), t = $(first(coherent_fit.t)):$(last(coherent_fit.t))")
end

main()
