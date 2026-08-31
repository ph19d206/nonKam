#!/usr/bin/env julia

"""
Plot the number-operator variance Var(n̂) as a function of time.

Run from the repository root:

    julia --project=. scripts/plot_var_n_vs_time.jl

For example, to use a log-log plot with custom parameters:

    N=5000 T_MAX=500 N0=0 K=0.5 R=2 SCALE=loglog \
        julia --project=. scripts/plot_var_n_vs_time.jl

Environment variables: N, T_MAX, N0, K, R, OMEGA, X0, P0, and SCALE.
ω defaults to 2/π and is fixed; R controls the kicking period τ = 2π/(ω*R).
SCALE defaults to loglog and may be linear, loglog, semilogx, or semilogy.
The data samples are at t = 0, ..., t_max - 1 and the resulting PDF is written to
plot/var_n_vs_time.pdf.
"""

using nonKam
using PythonPlot

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

env_number(name, default, type) = parse(type, get(ENV, name, string(default)))

# Kicked-harmonic-oscillator parameters.
# ω is held fixed (default ω = 2/π); R sets the kicking period via
# τ = 2π / (ω * R).  Changing R changes only τ, never ω.
R = env_number("R", 2, Int)
ω = env_number("OMEGA", 2 / π, Float64)
τ = 2π / (ω * R)
K = env_number("K", 0.5, Float64)

# Simulation and initial-state parameters
N = env_number("N", 3000, Int)
t_max = env_number("T_MAX", 1000, Int)
n = env_number("N0", 0, Int)
x0 = env_number("X0", -0.5903, Float64)
p0 = env_number("P0", -8.1437, Float64)
scale = Symbol(get(ENV, "SCALE", "loglog"))

N > 0 || throw(ArgumentError("N must be positive"))
t_max > 0 || throw(ArgumentError("T_MAX must be positive"))
0 <= n < N || throw(ArgumentError("N0 must satisfy 0 <= N0 < N"))

# Each entry is Var(n̂) before the corresponding evolution step.
variance_number = var_n(N, ω, τ, K, t_max, n)
variance_coherent = var_n(N, ω, τ, K, t_max, (x0, p0))
time = collect(0:(t_max - 1))

fig, ax = plt.subplots(figsize=(6, 4.5))
apply_publication_style!(ax; scale)

number_label = n == 0 ? raw"$|0\rangle$" : "\$|$(n)\\rangle\$"

"""Plot only positive coordinates when either axis has logarithmic scaling."""
function plot_variance!(values; color, label)
    log_x = scale in (:loglog, :semilogx)
    log_y = scale in (:loglog, :semilogy)
    time_mask = log_x ? time .> 0 : trues(length(time))
    value_mask = log_y ? values .> 0 : trues(length(values))
    mask = time_mask .& value_mask
    any(mask) || throw(ArgumentError("no positive variance data available for $scale scaling"))
    ax.plot(time[mask], values[mask]; color, linestyle="-", linewidth=1.4,
            marker=".", markersize=2.5, alpha=0.85, label)
end

plot_variance!(variance_number; color=PLOT_COLORS.blue, label=number_label)
plot_variance!(variance_coherent; color=PLOT_COLORS.vermillion, label=raw"$|\alpha\rangle$")

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$\mathrm{Var}(\hat{n})$")
parameter_text = "K = $(K)\nR = $(R)\n\$\\omega = $(round(ω, sigdigits=5))\$"
ax.text(0.03, 0.97, parameter_text; transform=ax.transAxes, ha="left", va="top",
        fontsize=8, bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                              "edgecolor" => "0.65", "alpha" => 0.9))
ax.legend(loc="best", frameon=false)

fig.tight_layout()
output_dir = joinpath(@__DIR__, "..", "plot")
mkpath(output_dir)
output_file = joinpath(output_dir, "var_n_vs_time.pdf")
save_publication_figure(fig, output_file)
PythonPlot.plotclose()

println("Saved number-operator variance plot to ", normpath(output_file))
