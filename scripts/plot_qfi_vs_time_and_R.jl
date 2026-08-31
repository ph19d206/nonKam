#!/usr/bin/env julia

"""
Plot QFI versus time for selected R values (left panel) and QFI versus R at
selected Floquet steps (right panel), in a single figure with one shared
log-scale QFI axis.

Data source: `data/qfi_time_R_sweep.jld2` (written by
scripts/qfi_time_R_sweep.jl).

Run from the repository root with:

    julia --project=. scripts/plot_qfi_vs_time_and_R.jl

or, from a Julia REPL:

    julia> include("scripts/plot_qfi_vs_time_and_R.jl")

Left panel  -- QFI vs physical time:
  * One curve per requested R value, drawn at the nearest stored R.
    The plotted-data window is restricted to TIME_RANGE (t_lo, t_hi);
    values are given in script units (physical time) and the same setting
    applies to every curve.
  * No fitting of any kind is performed.

Right panel -- QFI vs R at selected Floquet steps:
  * One curve per requested TIME_STEPS entry.  The physical time of a step
    is tau * step with tau = 2*pi/(omega*R) varying across the sweep, so the
    legend keeps the exact integer step with the symbolic tau: t = step . tau.

The y axis is shared between the panels (sharey=true), so the y-axis label
and tick labels appear only on the left panel.

Environment variables (each overrides the matching setting):
    INPUT_FILE    JLD2 data archive to load
                  default: data/qfi_time_R_sweep.jld2
    OUTPUT_FILE   plot file to write
                  default: plot/qfi_vs_time_and_R.pdf
    DATASET       1-based index of the archive entry to plot
    N             Fock-space dimension N of the entry to plot.  When set it
                  takes precedence over DATASET.
    T_MAX_LEFT    t_max of the left (time) panel's dataset.  When set it
                  takes precedence over N_LEFT; the archive can hold distinct
                  entries with equal N and different t_max.
    T_MAX_RIGHT   t_max of the right (R) panel's dataset.  Takes precedence
                  over N_RIGHT, mirroring T_MAX_LEFT.
    STATE         "Fock" or "Coherent" -- which initial state to plot
    R_VALUES      comma-separated list of R values for the left panel, e.g.
                  "1.9,1.95,2.0,2.05"
    TIME_RANGE    plotted-data time window for the left panel as "lo,hi",
                  e.g. "0.0,300.0".  Only data points with lo <= t <= hi are
                  drawn.
    TIME_STEPS    comma-separated list of Floquet steps for the right panel,
                  e.g. "20,100,500,1000"
    SHOW_LINE_M   "1"/"true" to draw the ~t^m guide line (default true)
    SHOW_LINE_N   "1"/"true" to draw the ~t^n guide line (default true)

Guide lines (left panel): dashed straight segments on the log-log axes,
anchored at (T_START_X, Y_AT_START_X) with slope X_POWER, sampled densely in
log space so each segment is straight on the log-log axes.  They are not
included in the legend; a small labelled text box (~t^m / ~t^n) is drawn
next to each line.  Set SHOW_LINE_M/SHOW_LINE_N = false to omit.

The defaults for all of the above live in the `USER SETTINGS` block at the
top of `main()` and can be edited directly in the file.
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using PythonPlot
using Statistics

using nonKam: plot_curves

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

"""Parse an environment variable as a boolean flag, with a default."""
function env_bool(name::AbstractString, default::Bool)
    value = lowercase(get(ENV, name, string(default)))
    return value in ("true", "1", "yes", "y")
end

"""Parse a comma-separated list of Float64 env-var values, with a default."""
function parse_float_list(env_name, default::Vector{Float64})
    text = strip(get(ENV, env_name, ""))
    isempty(text) && return default
    values = parse.(Float64, split(text, ','))
    isempty(values) && throw(ArgumentError("$env_name must contain at least one value"))
    return values
end

"""Parse the TIME_STEPS environment variable, falling back to `default`."""
function parse_time_steps(default::Vector{Int})
    text = get(ENV, "TIME_STEPS", "")
    isempty(strip(text)) && return default
    steps = parse.(Int, split(text, ','))
    isempty(steps) && throw(ArgumentError("TIME_STEPS must contain at least one step"))
    return steps
end

"""Resolve the plotted-data time window for the left panel.

`default` is the value from the USER SETTINGS block (`nothing` or a
`(lo, hi)` tuple).  An env var TIME_RANGE="lo,hi" overrides it.
Returns `nothing` when no window was requested.
"""
function resolve_time_range(default)
    text = strip(get(ENV, "TIME_RANGE", ""))
    if isempty(text)
        return default
    end
    parts = parse.(Float64, split(text, ','))
    length(parts) == 2 ||
        throw(ArgumentError("TIME_RANGE must be \"lo,hi\"; got '$text'"))
    parts[1] < parts[2] ||
        throw(ArgumentError("TIME_RANGE must satisfy lo < hi; got '$text'"))
    return (parts[1], parts[2])
end

function main()
    # =====================  USER SETTINGS  =====================
    # Edit the values below directly in this file.  Any of them can still be
    # overridden from the environment (e.g. ENV["R_VALUES"]="1.9,1.95,2.05")
    # -- the script reads the env var first and falls back to these values.
    # Each panel plots its own dataset from the archive.  N_LEFT picks the
    # entry for the time panel, N_RIGHT the entry for the R panel.
    N_LEFT     = 25000      # Fock-space dimension N of the left (time) dataset
    N_RIGHT    = 26000      # Fock-space dimension N of the right (R) dataset
    T_MAX_LEFT   = nothing  # t_max of the left (time) dataset; `nothing` uses N_LEFT
    T_MAX_RIGHT  = 200  # t_max of the right (R) dataset; `nothing` uses N_RIGHT
    STATE      = "Fock"     # "Fock" or "Coherent"

    # Left panel
    R_VALUES   = [2-1e-2, 2-1e-3, 2, 2+1e-3, 2+1e-2]   # R values (nearest stored R is used)
    TIME_RANGE = (9, 800)   # (t_lo, t_hi) plotted-data window; `nothing` = plot the full time range

    # Right panel
    TIME_STEPS = reverse([80, 100, 125, 160, 200])   # Floquet steps to draw (nearest stored step is used)
    # TIME_STEPS = reverse([40, 50, 80, 100, 125])


    # Guide lines ~t^m / ~t^n (left panel, log-log axes).  Optional dashed
    # straight segments anchored at (T_START_X, Y_AT_START_X) with exponent
    # X_POWER.  Not in the legend; labeled next to the line.
    SHOW_LINE_M = true
    T_START_M   = 80.0     # start t coordinate (line is anchored here)
    T_END_M     = 800.0    # end t coordinate
    Y_AT_START_M = 10^6.5  # y coordinate of the line at t = T_START_M
    M_POWER     = 6        # exponent of the t^m line

    SHOW_LINE_N = true
    T_START_N   = 200
    T_END_N     = 800
    Y_AT_START_N = 1.0e7
    N_POWER     = 2        # exponent of the t^n line

    TICK_LABELSIZE     = 14     # font size of the numeric tick labels on both axes
    AXIS_LABELSIZE     = 18     # font size of the x/y axis labels
    LEGEND_FONTSIZE    = 16     # font size of the legend text
    TEXT_FONTSIZE      = 20     # font size of the parameter text inside the plot
    CURVE_LINEWIDTH    = 3      # line width of every data curve (plot_curves default)

    OUTPUT_FILE        = "qfi_vs_time_and_R.pdf"   # plot file name inside plot/
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

    # Locate the archive entry with a given stored N (throwing with a helpful
    # message listing the available N values when it is missing).
    function find_dataset_by_N(target_N)
        hit = findfirst(ds -> ds.N == target_N, datasets)
        isnothing(hit) && throw(ArgumentError(
            "no dataset with N = $target_N in the archive. " *
            "Available N values: $(sort(unique([ds.N for ds in datasets])))."))
        return datasets[hit]
    end

    # Locate the archive entry with a given stored t_max (throwing with a
    # helpful message listing the available t_max values when it is missing).
    function find_dataset_by_t_max(target_t_max)
        hit = findfirst(ds -> ds.t_max == target_t_max, datasets)
        isnothing(hit) && throw(ArgumentError(
            "no dataset with t_max = $target_t_max in the archive. " *
            "Available t_max values: $(sort(unique([ds.t_max for ds in datasets])))."))
        return datasets[hit]
    end

    # Read a panel's N value from the env var (e.g. N_LEFT) or fall back to
    # the matching USER SETTINGS default.
    get_panel_N(name, default) =
        isempty(strip(get(ENV, name, ""))) ? default : parse(Int, get(ENV, name))

    # Resolve one panel's dataset.  An env var T_MAX_LEFT / T_MAX_RIGHT
    # (then the USER SETTINGS T_MAX_LEFT / T_MAX_RIGHT) takes precedence over
    # N: the archive holds distinct entries for equal N but different t_max,
    # so T_MAX pinpoints the exact one.  When unset, fall back to the N-based
    # N_LEFT / N_RIGHT lookup.
    function resolve_dataset(prefix, tmax_setting, n_default)
        tmax_text = strip(get(ENV, "T_MAX_$prefix", ""))
        if !isempty(tmax_text)
            return find_dataset_by_t_max(parse(Float64, tmax_text))
        end
        if !isnothing(tmax_setting)
            return find_dataset_by_t_max(tmax_setting)
        end
        return find_dataset_by_N(get_panel_N("N_$prefix", n_default))
    end
    dataset_left  = resolve_dataset("LEFT",  T_MAX_LEFT,  N_LEFT)
    dataset_right = resolve_dataset("RIGHT", T_MAX_RIGHT, N_RIGHT)

    # STATE selects "Fock" or "Coherent" inside an entry's qfi dictionary.
    state_name = get(ENV, "STATE", STATE)
    state_name = first(uppercase(state_name)) * lowercase(state_name[2:end])   # e.g. "fock" -> "Fock"
    function pick_run(dataset)
        haskey(dataset.qfi, state_name) ||
            throw(ArgumentError("STATE '$state_name' not among $(collect(keys(dataset.qfi)))."))
        return dataset.qfi[state_name]
    end
    run_left  = pick_run(dataset_left)
    run_right = pick_run(dataset_right)

    # Left panel (time) datasets
    qfiL, RL, timesL, time_stepsL = run_left.qfi, vec(run_left.R), run_left.times, vec(run_left.time_steps)
    # Right panel (R) datasets
    qfiR, RR, timesR, time_stepsR = run_right.qfi, vec(run_right.R), run_right.times, vec(run_right.time_steps)

    println("Left  (time) panel: $state_name state, (N, K, omega, eps, t_max) = " *
            "($(dataset_left.N), $(dataset_left.K), $(dataset_left.omega), " *
            "$(dataset_left.epsilon), $(dataset_left.t_max))")
    println("Right (R)    panel: $state_name state, (N, K, omega, eps, t_max) = " *
            "($(dataset_right.N), $(dataset_right.K), $(dataset_right.omega), " *
            "$(dataset_right.epsilon), $(dataset_right.t_max))")

    # Validation checks (left panel data used for the time plot)
    size(qfiL, 2) == length(RL) || throw(DimensionMismatch("qfiL columns must match length of RL"))
    size(qfiL, 1) == length(time_stepsL) || throw(DimensionMismatch("qfiL rows must match time_stepsL"))
    size(timesL) == size(qfiL) || throw(DimensionMismatch("timesL must have the same shape as qfiL"))

    # ------------------------------------------------------------------
    # User settings / env overrides for plotting
    # ------------------------------------------------------------------
    requested_r = parse_float_list("R_VALUES", R_VALUES)
    selected_steps = parse_time_steps(TIME_STEPS)
    time_range = resolve_time_range(TIME_RANGE)
    show_line_m = env_bool("SHOW_LINE_M", SHOW_LINE_M)
    show_line_n = env_bool("SHOW_LINE_N", SHOW_LINE_N)

    # Map requested R values to the nearest available column index (left panel
    # column indices come from the left dataset's R grid, which we also use for
    # the right panel's x values since both datasets share the same R sweep).
    selected_indices = [argmin(abs.(RL .- target_r)) for target_r in requested_r]

    # Map requested Floquet steps to the nearest available row index within
    # each panel's own dataset separately (the two datasets may have different
    # time grids).
    step_to_row(ds) = Dict(step => row for (row, step) in enumerate(vec(ds.time_steps)))
    map_steps(ds) = begin
        all_steps = sort(unique(vec(ds.time_steps)))
        map = step_to_row(ds)
        [map[argmin(abs.(all_steps .- target_step))] for target_step in selected_steps]
    end
    # The left panel plots whole R-columns (no step), so only the right panel
    # needs its steps mapped to row indices.
    selected_rows_right = map_steps(run_right)

    # --- One figure, two side-by-side subplots (independent y ranges) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5))
    # Apply the publication style to both axes (log-log left, linear-x right),
    # then override the font sizes from the USER SETTINGS block.
    apply_publication_style!(axL; scale=:loglog)
    apply_publication_style!(axR; scale=:semilogy)
    mpl.rcParams["xtick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["ytick.labelsize"] = TICK_LABELSIZE
    mpl.rcParams["axes.labelsize"] = AXIS_LABELSIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONTSIZE

    # -------------------------------------------------------------
    # LEFT PANEL: QFI vs physical time for selected R values
    # -------------------------------------------------------------
    x_left = Vector{Vector{Float64}}()
    y_left = Vector{Vector{Float64}}()
    for col in selected_indices
        t_vals = vec(timesL[:, col])
        qfi_vals = vec(qfiL[:, col])

        # Restrict the drawn points to the TIME_RANGE window (if any).
        t_plot = t_vals
        qfi_plot = qfi_vals
        if time_range !== nothing
            lo, hi = time_range
            keep = (t_vals .>= lo) .& (t_vals .<= hi)
            t_plot = t_vals[keep]
            qfi_plot = qfi_vals[keep]
        end

        push!(x_left, t_plot)
        push!(y_left, qfi_plot)
    end

    # Smart legend: an invisible (empty) first entry acts as the "R" header,
    # with the actual R values listed below it.  The header line is drawn
    # BEFORE the data so it appears on top of the legend, and plot_curves is
    # told not to build a legend itself -- the legend is constructed below.
    axL.plot([], []; color="none", label=raw"$R$")
    plot_curves(x_left, y_left,
                ax=axL,
                xlabel=raw"$t$",
                ylabel=raw"$\mathcal{I}(\omega; \vert \psi \rangle)$",
                labels=[string(round(RL[col], sigdigits=4))
                        for col in selected_indices],
                linewidth=CURVE_LINEWIDTH,
                legend=false)
    legL = axL.legend(fontsize=LEGEND_FONTSIZE)
    # Increase only the left panel's x-label font by +2 (plot_curves' default
    # label_fontsize is 20); the rest of the styling is left unchanged.
    axL.xaxis.label.set_fontsize(24)

    # -------------------------------------------------------------
    # RIGHT PANEL: QFI vs R at selected Floquet steps
    # -------------------------------------------------------------
    x_right = Vector{Vector{Float64}}()
    y_right = Vector{Vector{Float64}}()
    labels_right = Vector{String}()
    for (target_step, row) in zip(selected_steps, selected_rows_right)
        # Each curve is one Floquet step.  The physical time of that step is
        # tau * step, and tau = 2*pi/(omega*R) varies across the sweep, so the
        # physical time is not a single number for the whole curve.  The legend
        # therefore keeps only the exact integer step with the symbolic tau.
        # label = raw"$" * string(target_step) * raw"\,\tau$"
        label = raw"$" * string(target_step) * raw"$"
        push!(x_right, collect(RR))
        push!(y_right, vec(qfiR[row, :]))
        push!(labels_right, label)
    end
    # Smart legend for the right panel: an invisible first panel "t" header,
    # followed by the actual Floquet steps below it.  legend is built manually.
    axR.plot([], []; color="none", label=raw"$t$")
    plot_curves(x_right, y_right,
                ax=axR,
                xlabel=raw"$R$",
                # ylabel=raw"$\mathcal{I}(\mathbb{Q}; \sqrt{\psi} \rangle)$",
                labels=labels_right,
                linewidth=CURVE_LINEWIDTH,
                legend=false)
    legR = axR.legend(fontsize=LEGEND_FONTSIZE+2)

    # Right panel x-axis: the R range is a narrow sweep about the center, so
    # place exactly three integer R ticks (start, middle, end) and label them
    # with their rounded R values to avoid the dense default labelling.
    R_low, R_hi = minimum(RR), maximum(RR)
    axR.set_xticks(collect((R_low, (R_low + R_hi) / 2, R_hi)))
    axR.set_xticklabels([string(round(x; digits=5)) for x in (R_low, (R_low + R_hi) / 2, R_hi)])

    # axR.grid(true)
    # -------------------------------------------------------------
    # Extra annotations / styling (both panels)
    # -------------------------------------------------------------

    # Guide lines ~t^m / ~t^n on the LEFT panel (log-log axes).
    guides = Tuple[]
    show_line_m && push!(guides, (:m, T_START_M, T_END_M, Y_AT_START_M, M_POWER))
    show_line_n && push!(guides, (:n, T_START_N, T_END_N, Y_AT_START_N, N_POWER))
    for guide in guides
        _, t0, t1, y0, p = guide
        t0 >= t1 && error("guide line needs T_START < T_END (got $t0 >= $t1).")
        t0 > 0 || error("guide line needs T_START > 0 (got $t0).")
        y0 > 0 || error("guide line needs Y_AT_START > 0 (got $y0).")
        # Sample densely in log space so the segment is straight on log-log axes.
        tg = exp.(range(log(t0), log(t1); length=100))
        yg = y0 .* (tg ./ t0) .^ p
        axL.plot(tg, yg; color="0.3", linestyle="--", linewidth=1.6, zorder=5)
        # Label ~t^m / ~t^n at the midpoint of the segment, offset upward.
        tlab = sqrt(t0 * t1)
        ylab = y0 * (tlab / t0)^p
        power_char = guide[1] == :m ? M_POWER : N_POWER
        lbl = raw"$\sim t^{" * string(power_char) * raw"}$"
        axL.text(tlab*0.9, ylab * 1.5, lbl;
                 ha="center", va="bottom", color="0.3", fontsize=TEXT_FONTSIZE+3,
                 bbox=Dict("boxstyle" => "round,pad=0.2", "facecolor" => "white",
                           "edgecolor" => "none", "alpha" => 0.0))
    end

    # Panel labels (a) and (b) as the title of each subplot.
    axL.set_title(raw"(a)", fontsize=TEXT_FONTSIZE)
    axR.set_title(raw"(b)", fontsize=TEXT_FONTSIZE)

    # Parameter text (shared between both panels; drawn from the left dataset).
    parameter_text = "K = $(dataset_left.K)\n\$\\omega = $(round(dataset_left.omega, sigdigits=5))\$"
    axL.text(0.60, 0.30, parameter_text; transform=axL.transAxes, ha="left", va="top",
             fontsize=TEXT_FONTSIZE-2, bbox=Dict("boxstyle" => "round,pad=0.3",
                                               "facecolor" => "white",
                                               "edgecolor" => "0.65", "alpha" => 0.9))

    axR.text(0.05, 0.92, parameter_text; transform=axR.transAxes, ha="left", va="top",
             fontsize=TEXT_FONTSIZE-2, bbox=Dict("boxstyle" => "round,pad=0.3",
                                               "facecolor" => "white",
                                               "edgecolor" => "0.65", "alpha" => 0.9))

    # Tight layout: the left panel carries the y-axis label, so the two
    # panels are already width-balanced; just tidy the spacing.
    fig.tight_layout()
    mkpath(dirname(output_file))
    save_publication_figure(fig, output_file)
    plotclose()

    println("Saved QFI-vs-time-and-R plot to ", normpath(output_file))
    return nothing
end

main()