#!/usr/bin/env julia

"""
Plot the mean oscillator energy ⟨n̂⟩ and its intermediate power-law fit from a
previously computed JLD2 archive, without rerunning the simulation.

First generate the data once with:

    julia --project=. scripts/compute_mean_energy.jl

This writes data/mean_energy_vs_time.jld2. Then, to change only the plotting
style, rerun this script, for example:

    SCALE=linear OUTPUT_FILE=plot/mean_energy_vs_time_linear.pdf \
        julia --project=. scripts/plot_mean_energy_vs_time.jl

Environment variables:
    INPUT_FILE   JLD2 data archive to load
                 default: data/mean_energy_vs_time.jld2
    OUTPUT_FILE  base name for the plot file to write
                 default: plot/mean_energy_vs_time.pdf
    SCALE        linear, loglog, semilogx, or semilogy
                 default: scale stored in the JLD2 archive, or loglog

The `signatures` array below lists the parameter sets to plot.  Only datasets
in the archive whose `signature` matches one of these named tuples are
plotted; everything else is skipped.  Every plot is written to OUTPUT_FILE
(default: plot/mean_energy_vs_time.pdf), with no parameter details in the
name.  Edit the `signatures` list to choose what to plot.

`PLOT_FIT` (below) controls whether the power-law fits are drawn on top of the
data.  Set it to `false` to suppress the fit curves while the fit exponents m
are still shown in the corner text.  The fitted m values are always available
from the archive either way.

The fit exponents m and the fitting windows come from the archive, so only the
data computed in scripts/compute_mean_energy.jl is used here.
"""

# ---------------------------------------------------------------------------
# Choose which parameter sets to plot: list their signatures here.
#   signatures = [
#       (N = 1000, R = 2,  omega = 0.6366197723675814, x0 = 9.43687796076938, p0 = 0.0121, n = 0),
#       (N = 25000, R = 2, omega = 0.6366197723675814, x0 = 9.43687796076938, p0 = 0.0121, n = 0),
#   ]
# Only datasets matching one of these signatures are plotted.
# ---------------------------------------------------------------------------
const signatures = [
    (N = 25000, R = 2, omega = 2 / π, x0 = 0.0121 + 3 * pi, p0 = 0.0121, n = 0),
]

# Set to `false` to draw the data without the fitted power-law curves (the
# fit exponents m are still shown in the corner text).
const PLOT_FIT = false

# Font size of the numeric tick labels along the x (time) and y (energy) axes.
# The shared publication style uses 8; raise this to make the level markings
# bigger.
const TICK_LABELSIZE = 12

# X-axis (time) range of the plot.  Set to a tuple, e.g. X_LIMITS = (0.0, 200.0),
# to zoom to that time window; set to `nothing` to let the axis auto-scale.
const X_LIMITS = (0.0,300)

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

data_dir = joinpath(@__DIR__, "..", "data")
plot_dir = joinpath(@__DIR__, "..", "plot")

const default_output_file = joinpath(plot_dir, "mean_energy_vs_time.pdf")

function find_input_file()
    if haskey(ENV, "INPUT_FILE")
        f = ENV["INPUT_FILE"]
        isfile(f) || throw(ArgumentError("input file not found: $(normpath(f))"))
        return f
    end
    isdir(data_dir) || throw(ArgumentError(
        "data directory not found: $(normpath(data_dir)). Run scripts/compute_mean_energy.jl first."))
    f = joinpath(data_dir, "mean_energy_vs_time.jld2")
    isfile(f) && return f
    throw(ArgumentError("no data/mean_energy_vs_time.jld2 found under $(normpath(data_dir)). Run scripts/compute_mean_energy.jl first."))
end

"""Return true if the dataset's signature matches the requested named tuple."""
function matches_signature(ds, sig)
    ds.signature == sig
end

"""Build the plot for one dataset and save it to `file`.

`plot_fit` controls whether the fitted power-law curves are drawn.  The corner
text always reports the fit exponents m (from the archive) regardless of
`plot_fit`.  `x_limits` (a `(lo, hi)` tuple, or `nothing` for autoscaling)
clips the x-axis time window.
"""
function save_plot(data, file; scale, plot_fit, x_limits=nothing)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    apply_publication_style!(ax; scale)

    n = data.n
    number_label = n == 0 ? raw"$|0\rangle$" : "\$|$(n)\\rangle\$"

    ax.plot(data.time, data.energy_number; linestyle="-",
            marker="o", markersize=2.5, alpha=0.6, zorder=1, label=number_label)
    ax.plot(data.time, data.energy_coherent; color=PLOT_COLORS.vermillion, linestyle="-",
            marker="s", markersize=2.5, alpha=0.6, zorder=1, label=raw"$|\alpha\rangle$")

    # Only draw the fitted power-law curves when requested; the fit exponent m
    # is still reported in the corner text either way.
    fit_text = ""
    if plot_fit
        # A white underlay separates a fit from the data trace it necessarily overlaps.
        for (fit, color, label) in ((data.number_fit, PLOT_COLORS.blue, "number"),
                                    (data.coherent_fit, PLOT_COLORS.vermillion, "coherent"))
            ax.plot(fit.t, fit.energy; color="white", linewidth=4.0, zorder=3)
            ax.plot(fit.t, fit.energy; color=color, linestyle="--", linewidth=1.6, zorder=4,
                    label="$(label) fit: m = $(round(fit.m, digits=3)), t = $(first(fit.t)):$(last(fit.t))")
            ax.scatter([first(fit.t), last(fit.t)], [first(fit.energy), last(fit.energy)];
                       s=20, facecolors="white", edgecolors=color, linewidths=1.1, zorder=5)
        end
    else
        fit_text = "m_num = $(round(data.number_fit.m, digits=3))\n" *
                   "m_coh = $(round(data.coherent_fit.m, digits=3))"
    end

    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$\langle \hat{n} \rangle$")

    K = data.K
    R = data.R
    ω = data.omega
    parameter_text = "K = $(K)\nR = $(R)\n\$\\omega = $(round(ω, sigdigits=5))\$"
    if !isempty(fit_text)
        parameter_text *= "\n" * fit_text
    end
    ax.text(0.03, 0.97, parameter_text; transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                                  "edgecolor" => "0.65", "alpha" => 0.9))
    ax.legend(loc="lower right", frameon=false)

    if x_limits !== nothing
        ax.set_xlim(x_limits)
    end

    fig.tight_layout()
    mkpath(dirname(file))
    save_publication_figure(fig, file)
    PythonPlot.plotclose()
    return file
end

input_file = find_input_file()
base_output = get(ENV, "OUTPUT_FILE", default_output_file)

datasets = jldopen(input_file, "r") do file
    haskey(file, "datasets") ? Vector(file["datasets"]) : any[]
end

isempty(datasets) && throw(ArgumentError("the archive $(normpath(input_file)) holds no datasets."))

stored_scale = hasproperty(datasets[1], :scale) ? datasets[1].scale : :loglog
scale = Symbol(get(ENV, "SCALE", string(stored_scale)))

println("Loaded $(length(datasets)) dataset(s) from ", normpath(input_file))

plotted = String[]
for (i, sig) in enumerate(signatures)
    matched = filter(ds -> matches_signature(ds, sig), datasets)
    if isempty(matched)
        println("Signature $i: no dataset matches ", sig, " (skipped)")
        continue
    end
    for ds in matched
        push!(plotted, save_plot(ds, base_output; scale, plot_fit=PLOT_FIT,
                                 x_limits=X_LIMITS))
    end
end

if isempty(plotted)
    throw(ArgumentError("none of the listed signatures matched a dataset in " *
                        "$(normpath(input_file)); nothing plotted."))
end
for p in plotted
    println("Saved mean-energy plot to ", normpath(p))
end
