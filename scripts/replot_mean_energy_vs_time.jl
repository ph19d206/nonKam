#!/usr/bin/env julia

"""
Replot the saved mean oscillator energy data without rerunning the simulation.

First generate the data once with:

    julia --project=. scripts/plot_mean_energy_vs_time.jl

Then change only the plotting style and rerun this script, for example:

    SCALE=linear OUTPUT_FILE=plot/mean_energy_vs_time_linear.pdf \
        julia --project=. scripts/replot_mean_energy_vs_time.jl

Environment variables:
    INPUT_FILE   JLD2 data archive to load
                 default: plot/mean_energy_vs_time.jld2
    OUTPUT_FILE  plot file to write
                 default: plot/mean_energy_vs_time_replot.pdf
    SCALE        linear, loglog, semilogx, or semilogy
                 default: scale stored in the JLD2 archive, or loglog
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

const default_input_file = joinpath(@__DIR__, "..", "plot", "mean_energy_vs_time.jld2")
const default_output_file = joinpath(@__DIR__, "..", "plot", "mean_energy_vs_time_replot.pdf")

input_file = get(ENV, "INPUT_FILE", default_input_file)
output_file = get(ENV, "OUTPUT_FILE", default_output_file)

isfile(input_file) ||
    throw(ArgumentError("input file not found: $(normpath(input_file)). Run scripts/plot_mean_energy_vs_time.jl first."))

data = jldopen(input_file, "r") do file
    (
        time = file["time"],
        energy_number = file["energy_number"],
        energy_coherent = file["energy_coherent"],
        number_fit = file["number_fit"],
        coherent_fit = file["coherent_fit"],
        parameters = file["parameters"],
    )
end

stored_scale = hasproperty(data.parameters, :scale) ? data.parameters.scale : :loglog
scale = Symbol(get(ENV, "SCALE", string(stored_scale)))

fig, ax = plt.subplots(figsize=(6, 4.5))
apply_publication_style!(ax; scale)

n = data.parameters.n
number_label = n == 0 ? raw"$|0\rangle$" : "\$|$(n)\\rangle\$"

ax.plot(data.time, data.energy_number; color=PLOT_COLORS.blue, linestyle="None",
        marker=".", markersize=8, alpha=0.6, zorder=1, label=number_label)
ax.plot(data.time, data.energy_coherent; color=PLOT_COLORS.vermillion, linestyle="None",
        marker=".", markersize=8, alpha=0.6, zorder=1, label=raw"$|\alpha\rangle$")

# for (fit, color, label) in ((data.number_fit, PLOT_COLORS.blue, "number"),
#                             (data.coherent_fit, PLOT_COLORS.vermillion, "coherent"))
#     ax.plot(fit.t, fit.energy; color="white", linewidth=4.0, zorder=3)
#     ax.plot(fit.t, fit.energy; color=color, linestyle="--", linewidth=1.6, zorder=4,
#             label="$(label) fit: m = $(round(fit.m, digits=3)), t = $(first(fit.t)):$(last(fit.t))")
#     ax.scatter([first(fit.t), last(fit.t)], [first(fit.energy), last(fit.energy)];
#                s=20, facecolors="white", edgecolors=color, linewidths=1.1, zorder=5)
# end

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$\langle \hat{n} \rangle$")

K = data.parameters.K
R = data.parameters.R
ω = data.parameters.omega
parameter_text = "K = $(K)\nR = $(R)\n\$\\omega = $(round(ω, sigdigits=5))\$"
ax.text(0.03, 0.97, parameter_text; transform=ax.transAxes, ha="left", va="top",
        fontsize=8, bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                              "edgecolor" => "0.65", "alpha" => 0.9))
ax.legend(loc="lower right", frameon=false)

fig.tight_layout()
mkpath(dirname(output_file))
save_publication_figure(fig, output_file)
PythonPlot.plotclose()

println("Loaded mean-energy data from ", normpath(input_file))
println("Saved mean-energy replot to ", normpath(output_file))
