#!/usr/bin/env julia

"""
Plot QFI versus R for selected rows of the qfi R-sweep archive
(`data/qfi_time_R_sweep.jld2`, written by scripts/qfi_time_R_sweep.jl).

Run from the repository root with:

    julia --project=. scripts/plot_qfi_vs_R_at_times.jl

or, from a Julia REPL:

    julia> include("scripts/plot_qfi_vs_R_at_times.jl")

The default curves are time steps 20, 100, 500, and 1000.  They can be
overridden with, for example, `TIME_STEPS=1,250,750,1000`.

Environment variables:
    INPUT_FILE   JLD2 data archive to load
                 default: data/qfi_time_R_sweep.jld2
    OUTPUT_FILE  plot file to write
                 default: plot/qfi_vs_R_at_times.pdf
    DATASET      1-based index of the archive entry to plot (default 1)
    STATE        "Fock" (default) or "Coherent" -- which initial state
                 to plot from the entry's `qfi` dictionary
    TIME_STEPS   comma-separated list of Floquet steps to draw, e.g.
                 "1,250,750,1000"
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot
using Statistics

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

"""Parse the TIME_STEPS environment variable, falling back to `default`."""
function parse_time_steps(default::Vector{Int})
    text = get(ENV, "TIME_STEPS", "")
    isempty(strip(text)) && return default
    steps = parse.(Int, split(text, ','))
    isempty(steps) && throw(ArgumentError("TIME_STEPS must contain at least one step"))
    return steps
end

function main()
    # =====================  USER SETTINGS  =====================
    # Edit the values below directly in this file.  Any of them can still be
    # overridden from the environment (e.g. ENV["TIME_STEPS"]="1,250,750,1000")
    # -- the script reads the env var first and falls back to these values.
    DATASET    = 1          # 1-based index of the archive entry to plot (used only when N is not set)
    N          = nothing    # Fock-space dimension N of the entry to plot; `nothing` = use DATASET
    STATE      = "Fock"     # "Fock" or "Coherent"
    TIME_STEPS = [20, 100, 500, 1000]   # Floquet steps to draw (nearest stored step is used)
    TICK_LABELSIZE     = 14      # font size of the numeric tick labels on both axes
                                 # (the shared publication style uses 8)
    AXIS_LABELSIZE     = 18      # font size of the x/y axis labels
    LEGEND_FONTSIZE    = 12      # font size of the legend text
    TEXT_FONTSIZE      = 14      # font size of the parameter text inside the plot
    # ===========================================================

    data_dir = joinpath(@__DIR__, "..", "data")
    plot_dir = joinpath(@__DIR__, "..", "plot")

    input_file = get(ENV, "INPUT_FILE", joinpath(data_dir, "qfi_time_R_sweep.jld2"))
    output_file = get(ENV, "OUTPUT_FILE", joinpath(plot_dir, "qfi_vs_R_at_times.pdf"))

    isfile(input_file) ||
        throw(ArgumentError("input file not found: $(normpath(input_file)). " *
                            "Run scripts/qfi_time_R_sweep.jl first."))

    datasets = jldopen(input_file, "r") do file
        Vector(file["datasets"])
    end
    isempty(datasets) && throw(ArgumentError("the archive contains no datasets."))

    # DATASET picks which parameter set to plot; an env var can still override
    # the value from the USER SETTINGS block above.  When N is set (as a
    # setting or via the N environment variable), the entry whose stored N
    # matches is plotted instead, taking precedence over DATASET.
    n_str = get(ENV, "N", "")
    target_N = isempty(strip(n_str)) ? N : tryparse(Int, n_str)
    if target_N isa Int
        hit = findfirst(ds -> ds.N == target_N, datasets)
        isnothing(hit) && throw(ArgumentError(
            "no dataset with N = $target_N in the archive. " *
            "Available N values: $(sort(unique([ds.N for ds in datasets])))."))
        idx = hit
    else
        isnothing(target_N) || throw(ArgumentError("N must be an integer; got '$n_str'."))
        dataset_idx = get(ENV, "DATASET", string(DATASET))
        idx = tryparse(Int, dataset_idx)
        isnothing(idx) && throw(ArgumentError("DATASET must be an integer; got '$dataset_idx'."))
        1 <= idx <= length(datasets) ||
            throw(ArgumentError("DATASET index $idx out of range 1:$(length(datasets))."))
    end
    dataset = datasets[idx]

    # STATE selects "Fock" or "Coherent" inside the entry's qfi dictionary.
    state_name = get(ENV, "STATE", "Fock")
    haskey(dataset.qfi, state_name) ||
        throw(ArgumentError("STATE '$state_name' not among $(collect(keys(dataset.qfi)))."))

    run = dataset.qfi[state_name]
    qfi, R, times, time_steps = run.qfi, vec(run.R), run.times, vec(run.time_steps)

    size(qfi, 2) == length(R) || throw(DimensionMismatch("qfi columns must match R"))
    size(qfi, 1) == length(time_steps) || throw(DimensionMismatch("qfi rows must match time_steps"))
    size(times) == size(qfi) || throw(DimensionMismatch("times must have the same shape as qfi"))

    println("Plotting $state_name state, dataset $idx/$(length(datasets)): " *
            "(N, K, omega, eps, t_max) = ($(dataset.N), $(dataset.K), " *
            "$(dataset.omega), $(dataset.epsilon), $(dataset.t_max))")

    selected_steps = parse_time_steps([20, 100, 500, 1000])
    step_to_row = Dict(step => row for (row, step) in enumerate(time_steps))
    all(haskey(step_to_row, step) for step in selected_steps) ||
        throw(ArgumentError("requested TIME_STEPS must be present in the archive"))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    apply_publication_style!(ax; scale=:linear)
    mpl.rcParams["xtick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["ytick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["axes.labelsize"] = AXIS_LABELSIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONTSIZE

    colors = [PLOT_COLORS.blue, PLOT_COLORS.vermillion, PLOT_COLORS.green, PLOT_COLORS.purple]
    for (curve_index, step) in enumerate(selected_steps)
        row = step_to_row[step]
        # Each curve is one Floquet step.  The physical time of that step is
        # tau * step, and tau = 2*pi/(omega*R) varies across the sweep, so the
        # physical time is not a single number for the whole curve.  The legend
        # therefore keeps only the exact integer step with the symbolic tau:
        #   t = step * tau   (tau varies with R).
        label = raw"$t = " * string(step) * raw"\,\tau$"
        ax.plot(R, qfi[row, :]; color=colors[mod1(curve_index, length(colors))],
                linewidth=1.5, marker="o", markersize=2.5, label=label)
    end

    ax.set_xlabel(raw"$R$")
    ax.set_ylabel(raw"$\mathcal{I}(\omega; \vert \psi \rangle)$")
    # ax.set_xscale("log")
    ax.set_yscale("log")

    parameter_text = "K = $(dataset.K)\n\$\\omega = $(round(dataset.omega, sigdigits=5))\$"
    ax.text(0.03, 0.97, parameter_text; transform=ax.transAxes, ha="left", va="top",
            fontsize=TEXT_FONTSIZE, bbox=Dict("boxstyle" => "round,pad=0.3",
                                              "facecolor" => "white",
                                              "edgecolor" => "0.65", "alpha" => 0.9))

    ax.legend(loc="best", frameon=false, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    mkpath(dirname(output_file))
    save_publication_figure(fig, output_file)
    PythonPlot.plotclose()

    println("Saved QFI-versus-R plot to ", normpath(output_file))
    return nothing
end

main()