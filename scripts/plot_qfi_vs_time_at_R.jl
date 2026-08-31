#!/usr/bin/env julia

"""
Plot QFI versus physical time for selected R values from the qfi R-sweep
archive (`data/qfi_time_R_sweep.jld2`, written by scripts/qfi_time_R_sweep.jl).

Run from the repository root with:

    julia --project=. scripts/plot_qfi_vs_time_at_R.jl

or, from a Julia REPL:

    julia> include("scripts/plot_qfi_vs_time_at_R.jl")

The default R values are 1.90, 1.95, 2.00, and 2.05.  They can be overridden
with, for example, `R_VALUES=1.9,1.95,2.0,2.05`; each requested R is drawn
at the nearest R present in the archive.

For each drawn curve the script computes the intermediate power-law fit
F_Q(t) ≈ C * t^m in log--log space with `find_converged_m` (src/find_converged_m.jl),
the same tool the mean-energy scripts use, and annotates each curve with its
exponent m.  The fitted line is drawn only when SHOW_FITS=1 (see below).

Environment variables (each overrides the matching setting):
    INPUT_FILE   JLD2 data archive to load
                 default: data/qfi_time_R_sweep.jld2
    OUTPUT_FILE  plot file to write
                 default: plot/qfi_vs_time_at_R.pdf
    DATASET      1-based index of the archive entry to plot
    N            Fock-space dimension N of the entry to plot.  When set it
                 takes precedence over DATASET: the entry whose stored N
                 matches is plotted.  Useful when the archive holds several
                 datasets with different N.
    STATE        "Fock" or "Coherent" -- which initial state to plot
    R_VALUES     comma-separated list of R values to draw, e.g.
                 "1.9,1.95,2.0,2.05"
    SHOW_FITS    "1"/"true" to also draw the dashed power-law fits,
                 "0"/"false" to draw only the data
    TIME_RANGE   plotted-data time window as "lo,hi", e.g. "0.0,300.0".
                 Only data points with lo <= t <= hi are drawn; the fit is
                 still computed over the full stored range.
    FIT_MIN_POINTS  minimum number of positive data points required for a fit
    FIT_MIN_ABS_SLOPE  |m| below which the fit is treated as a saturation plateau
    FIT_TOL      relative tolerance on the local-log--log-slope scatter

The defaults for all of the above live in the `USER SETTINGS` block at the
top of `main()` and can be edited directly in the file.
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot
using Statistics
using nonKam: find_converged_m, plot_curves

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

"""Parse an environment variable as a boolean flag, with a default."""
function env_bool(name::AbstractString, default::Bool)
    value = lowercase(get(ENV, name, string(default)))
    return value in ("true", "1", "yes", "y")
end

"""Parse the R_VALUES environment variable, falling back to `default`."""
function parse_r_values(default::Vector{Float64})
    text = get(ENV, "R_VALUES", "")
    isempty(strip(text)) && return default
    values = parse.(Float64, split(text, ','))
    isempty(values) && throw(ArgumentError("R_VALUES must contain at least one numerical value"))
    return values
end

"""Resolve the plotted-data time window.

`default` is the value from the USER SETTINGS block (`nothing` or a
`(lo, hi)` tuple).  An env var TIME_RANGE="lo,hi" (or "lo,hi,step") overrides
it.  Returns `nothing` when no window was requested.
"""
function resolve_time_range(default)
    text = strip(get(ENV, "TIME_RANGE", ""))
    if isempty(text)
        return default
    end
    parts = parse.(Float64, split(text, ','))
    length(parts) == 2 || length(parts) == 3 ||
        throw(ArgumentError("TIME_RANGE must be \"lo,hi\" or \"lo,hi,step\"; got '$text'"))
    return (parts[1], parts[2])
end

function main()
    # =====================  USER SETTINGS  =====================
    # Edit the values below directly in this file.  Any of them can still be
    # overridden from the environment (e.g. ENV["SHOW_FITS"]="1") -- the
    # script reads the env var first and falls back to these values.
    DATASET    = 1          # 1-based index of the archive entry to plot (used only when N is not set)
    N          = 25000    # Fock-space dimension N of the entry to plot; `nothing` = use DATASET
    STATE      = "Fock"     # "Fock" or "Coherent"
    R_VALUES   = [2-1e-2,2-1e-3, 2, 2+1e-3,2+1e-2]   # R values to draw (nearest stored R is used)
    SHOW_FITS  = false       # draw the dashed power-law fits (true/false)
    TIME_RANGE = (9, 800)   # (t_lo, t_hi) plotted-data window, e.g. (0.0, 300.0);
                           # `nothing` = plot the full time range.  The fit is
                           # still computed over the full range (unchanged).
    FIT_MIN_POINTS     = 10       # minimum positive points required for a fit
    FIT_MIN_ABS_SLOPE  = 0.05    # |m| below which the fit counts as flat/saturated
    FIT_TOL            = 0.05    # tolerance on local log-log slope scatter
    TICK_LABELSIZE     = 14      # font size of the numeric tick labels on both axes
                                 # (the shared publication style uses 8)
    AXIS_LABELSIZE     = 18      # font size of the x/y axis labels
    LEGEND_FONTSIZE    = 12      # font size of the legend text
    TEXT_FONTSIZE      = 14      # font size of the parameter text inside the plot

    # Optional reference guide lines ~t^m and ~t^n.  Each line is a dashed
    # straight segment on the log-log axes from T_START_X to T_END_X, passing
    # through (T_START_X, Y_AT_START_X), i.e.
    #     y(t) = Y_AT_START_X * (t / T_START_X)^X_POWER
    # They are not included in the legend; the label ~t^m / ~t^n is drawn in a
    # small text box next to the line.  Set SHOW_LINE_X = false to omit.
    SHOW_LINE_M = true
    T_START_M   = 80.0     # start t coordinate (line is anchored here)
    T_END_M     = 800.0    # end t coordinate
    Y_AT_START_M = 10^6.5  # y coordinate of the line at t = T_START_M
    M_POWER     = 6     # exponent of the t^m line

    SHOW_LINE_N = true
    T_START_N   = 200
    T_END_N     = 800
    Y_AT_START_N = 1.0e7
    N_POWER     = 2
    OUTPUT_FILE        = "qfi_vs_time_at_R.pdf"   # plot file name inside plot/
    # ===========================================================

    data_dir = joinpath(@__DIR__, "..", "data")
    plot_dir = joinpath(@__DIR__, "..", "plot")

    input_file = get(ENV, "INPUT_FILE", joinpath(data_dir, "qfi_time_R_sweep.jld2"))
    output_file = get(ENV, "OUTPUT_FILE", joinpath(plot_dir, OUTPUT_FILE))

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
    state_name = get(ENV, "STATE", STATE)
    state_name = first(uppercase(state_name)) * lowercase(state_name[2:end])   # e.g. "fock" -> "Fock"
    haskey(dataset.qfi, state_name) ||
        throw(ArgumentError("STATE '$state_name' not among $(collect(keys(dataset.qfi)))."))

    run = dataset.qfi[state_name]
    qfi, R, times, time_steps = run.qfi, vec(run.R), run.times, vec(run.time_steps)

    println("Plotting $state_name state, dataset $idx/$(length(datasets)): " *
            "(N, K, omega, eps, t_max) = ($(dataset.N), $(dataset.K), " *
            "$(dataset.omega), $(dataset.epsilon), $(dataset.t_max))")

    # Validation checks
    size(qfi, 2) == length(R) || throw(DimensionMismatch("qfi columns must match length of R"))
    size(qfi, 1) == length(time_steps) || throw(DimensionMismatch("qfi rows must match time_steps"))
    size(times) == size(qfi) || throw(DimensionMismatch("times must have the same shape as qfi"))

    # Map requested R values to the nearest available column index in the dataset
    requested_r = parse_r_values(R_VALUES)
    selected_indices = [argmin(abs.(R .- target_r)) for target_r in requested_r]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    apply_publication_style!(ax; scale=:linear)
    mpl.rcParams["xtick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["ytick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["axes.labelsize"] = AXIS_LABELSIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONTSIZE

    colors = [PLOT_COLORS.blue, PLOT_COLORS.vermillion, PLOT_COLORS.green,
              PLOT_COLORS.purple, PLOT_COLORS.black]

    # Power-law fit knobs, forwarded to `find_converged_m` (transient_fraction
    # is intentionally left at its default of zero: the stability test decides).
    show_fits = env_bool("SHOW_FITS", SHOW_FITS)
    fit_min_points = parse(Int, get(ENV, "FIT_MIN_POINTS", string(FIT_MIN_POINTS)))
    fit_min_abs_slope = parse(Float64, get(ENV, "FIT_MIN_ABS_SLOPE", string(FIT_MIN_ABS_SLOPE)))
    fit_tol = parse(Float64, get(ENV, "FIT_TOL", string(FIT_TOL)))
    time_range = resolve_time_range(TIME_RANGE)

    for (curve_index, col) in enumerate(selected_indices)
        actual_R = R[col]

        # Extract corresponding physical time and QFI vector for column `col`
        t_vals = times[:, col]
        qfi_vals = qfi[:, col]

        # The plotted data is restricted to the TIME_RANGE window (if any).
        # The fit below still runs over the full t_vals/qfi_vals.
        t_plot = t_vals
        qfi_plot = qfi_vals
        if time_range !== nothing
            lo, hi = time_range
            keep = (t_vals .>= lo) .& (t_vals .<= hi)
            t_plot = t_vals[keep]
            qfi_plot = qfi_vals[keep]
        end

        # Fit F_Q(t) ~ C * t^m in log-log space; the exponent is annotated
        # on the legend even when the fitted line itself is not drawn.
        m_text = ""
        fit_line = nothing
        ok, fit = try
            f = find_converged_m(t_vals, qfi_vals;
                                 min_points=fit_min_points,
                                 min_abs_slope=fit_min_abs_slope,
                                 tol=fit_tol)
            (true, f)
        catch err
            (false, nothing)
        end
        if ok
            # Y-values for the fitted interval, reconstructed with a fresh
            # log-space linear regression on the same (t, F_Q) points that
            # `find_converged_m` regressed; m matches `fit.m_converged`.
            tk = fit.t_converged
            idx = [argmin(abs.(t_vals .- tk[k])) for k in eachindex(tk)]
            yk = qfi_vals[idx]
            xlog = log.(tk)
            ylog = log.(yk)
            A = hcat(xlog, ones(length(xlog)))
            m_fit, log_prefactor = A \ ylog
            fit_line = (t=tk, qfi_val=exp.(log_prefactor .+ m_fit .* xlog), m=m_fit)
            m_text = string(round(m_fit, digits=3))
        end
        # The legend shows only the R value; the corresponding exponent m is
        # printed to the console (see the println below).
        label = raw"$R = " * string(round(actual_R, sigdigits=4)) * raw"$"
        if ok
            println("R = $(round(actual_R, sigdigits=4))  ->  m = $m_text")
        end

        ax.plot(t_plot, qfi_plot;
                color=colors[mod1(curve_index, length(colors))],
                linewidth=1.5, label=label)

        if ok && show_fits
            # A white underlay separates the fit from the data trace it
            # necessarily overlaps (same convention as plot_mean_energy_vs_time.jl).
            ax.plot(fit_line.t, fit_line.qfi_val; color="white", linewidth=4.0, zorder=3)
            ax.plot(fit_line.t, fit_line.qfi_val; color=colors[mod1(curve_index, length(colors))],
                    linestyle="--", linewidth=1.6, zorder=4)
        end
    end

    # Optional reference guide lines ~t^m and ~t^n: dashed straight segments
    # on the log-log axes, anchored at (T_START_X, Y_AT_START_X) with slope
    # X_POWER.  Not in the legend; labeled next to the line instead.
    guides = Tuple[]
    SHOW_LINE_M && push!(guides, (:m, T_START_M, T_END_M, Y_AT_START_M, M_POWER))
    SHOW_LINE_N && push!(guides, (:n, T_START_N, T_END_N, Y_AT_START_N, N_POWER))
    for guide in guides
        _, t0, t1, y0, p = guide
        t0 >= t1 && error("guide line needs T_START < T_END (got $t0 >= $t1).")
        t0 > 0 || error("guide line needs T_START > 0 (got $t0).")
        y0 > 0 || error("guide line needs Y_AT_START > 0 (got $y0).")
        # Sample densely in log space so the segment is straight on log-log axes.
        tg = exp.(range(log(t0), log(t1); length=100))
        yg = y0 .* (tg ./ t0) .^ p
        ax.plot(tg, yg; color="0.3", linestyle="--", linewidth=1.6, zorder=5)
        # Label ~t^m / ~t^n at the midpoint of the segment, offset upward.
        tlab = sqrt(t0 * t1)
        ylab = y0 * (tlab / t0)^p
        power_char = guide[1] == :m ? "$M_POWER" : "$N_POWER"
        lbl = "\$\\sim t^{" * power_char * "}\$"
        ax.text(tlab, ylab * 1.4, lbl;
                ha="center", va="bottom", color="0.3", fontsize=TEXT_FONTSIZE,
                bbox=Dict("boxstyle" => "round,pad=0.2", "facecolor" => "white",
                          "edgecolor" => "none", "alpha" => 0.8))
    end

    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$\mathcal{I}(\omega; \vert \psi \rangle)$")
    ax.set_xscale("log")
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

    println("Saved QFI-versus-time plot to ", normpath(output_file))
    return nothing
end

main()