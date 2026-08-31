#!/usr/bin/env julia

"""
Plot the per-rotation-angle ratio CFI[ϕ] / QFI for precomputed single-time
datasets, without rerunning the simulation.

First generate the data once with:

    julia --project=. scripts/compute_cfi_qfi_ratio_single_time.jl

This appends datasets to data/cfi_qfi_ratio_single_time.jld2 (key "datasets").
Each dataset carries `phi` (rotation-angle grid), `cfi` [nϕ,1], `qfi`
(length-1), and `ratio` [nϕ,1] = cfi ./ qfi. This plotting script loads every
stored dataset and draws the ratio curve vs ϕ, labelling each curve by its time
step (and the parameter identity K, R, ω, ε).

To change only the plotting style rerun, for example:

    INPUT_FILE=data/cfi_qfi_ratio_single_time.jld2 \
    OUTPUT_FILE=plot/cfi_qfi_ratio_single_time.pdf \
        julia --project=. scripts/plot_cfi_qfi_ratio_single_time.jl

Environment variables:
    INPUT_FILE   JLD2 data archive to load
                 default: data/cfi_qfi_ratio_single_time.jld2
    OUTPUT_FILE  plot file to write
                 default: plot/cfi_qfi_ratio_single_time.pdf
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

data_dir = joinpath(@__DIR__, "..", "data")
plot_dir = joinpath(@__DIR__, "..", "plot")

const default_input_file = joinpath(data_dir, "cfi_qfi_ratio_single_time.jld2")
const default_output_file = joinpath(plot_dir, "cfi_qfi_ratio_single_time.pdf")

input_file = get(ENV, "INPUT_FILE", default_input_file)
output_file = get(ENV, "OUTPUT_FILE", default_output_file)

isfile(input_file) ||
    throw(ArgumentError("input file not found: $(normpath(input_file)). Run " *
                        "scripts/compute_cfi_qfi_ratio_single_time.jl first."))

# ---- Load every stored dataset ----------------------------------------
datasets = jldopen(input_file, "r") do f
    haskey(f, "datasets") ? Vector(f["datasets"]) : Any[]
end

isempty(datasets) &&
    throw(ArgumentError("no datasets found in $(normpath(input_file))."))

# ---- Plot -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
apply_publication_style!(ax; scale=:linear)

# Keep the axis linear rather than logging, since the ratio spans (and can
# cross) negative values where a log scale is undefined.
ax.axhline(1.0; color=PLOT_COLORS.black, linestyle=":", linewidth=0.8, zorder=1,
           label="CFI = QFI")

for ds in sort(datasets; by=ds -> ds.time_steps[1])
    phi  = vec([Float64(x) for x in ds.phi])
    ratio = vec([Float64(x) for x in ds.ratio])

    label = "t = $(ds.time_steps[1])"
    ax.plot(phi, ratio; color=PLOT_COLORS.blue, linestyle="-", linewidth=1.2,
            zorder=2, label=label)
end

ax.set_xlabel(raw"rotation angle $\phi$")
ax.set_ylabel(raw"$CFI(\phi)/QFI$")

# ---- Parameter annotation from the (last) dataset ---------------------
ds = datasets[end]
K = ds.K; R = ds.R; ω = ds.omega; ϵ = ds.epsilon
parameter_text = "N = $(ds.N)\nK = $(K)\nR = $(R)\n\$\\omega = $(round(ω, sigdigits=5))\$" *
                 "\n\$\\epsilon = $(round(ϵ, sigdigits=3))\$"
ax.text(0.03, 0.97, parameter_text; transform=ax.transAxes, ha="left", va="top",
        fontsize=8, bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                              "edgecolor" => "0.65", "alpha" => 0.9))
ax.legend(loc="best", frameon=false)

fig.tight_layout()
mkpath(dirname(output_file))
save_publication_figure(fig, output_file)
PythonPlot.plotclose()

println("Loaded $(length(datasets)) dataset(s) from ", normpath(input_file))
println("Saved CFI/QFI ratio plot to ", normpath(output_file))