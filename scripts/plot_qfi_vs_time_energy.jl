#!/usr/bin/env julia

"""
Two-panel plot: 
  * Left  = QFI (Fisher information) vs time for a Fock state (|n0⟩) and a coherent state (|α⟩)
  * Right = Mean energy ⟨n̂⟩ vs time for the same two states.

Both panels are queried for the same chosen physical simulation parameters:
  R, K, N, ω, n0, (x0, p0).

Data sources (both JLD2):
  * LEFT  — scripts/calculate_qfi_time_states.jl -> data/qfi_simulations.jld2
  * RIGHT — scripts/compute_mean_energy.jl       -> data/mean_energy_vs_time.jld2

Run (from the repository root):

    julia --project=. scripts/plot_qfi_vs_time_energy.jl

Or with custom parameters via environment variables:

    R=4.0 K=1.5 N=25000 OMEGA=0.63662 N0=0 X0=-0.5903 P0=-8.1437 TMIN=0 TMAX=500 julia --project=. scripts/plot_qfi_vs_time_energy.jl

Environment variables:
    R                   Floquet period parameter (default: 4.0)
    K                   Kicking strength (default: 1.5)
    N                   Hilbert space dimension (default: 25000)
    OMEGA               Oscillator frequency (default: 2/π)
    N0                  Fock / number state index (default: 0)
    X0                  Coherent state coordinate x0 (default: -0.5903)
    P0                  Coherent state coordinate p0 (default: -8.1437)
    TMIN                Minimum time for QFI plot (default: nothing, full range)
    TMAX                Maximum time for QFI plot (default: nothing, full range)
    INPUT_FILE          JLD2 archive for QFI (default: data/qfi_simulations.jld2)
    ENERGY_INPUT_FILE   JLD2 archive for Mean Energy (default: data/mean_energy_vs_time.jld2)
    OUTPUT_FILE         Plot file to write (default: plot/qfi_vs_time_energy.pdf)
"""

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using JLD2
using Statistics
using PythonPlot
using nonKam: plot_curves

const plt = PythonPlot
const mpl = PythonPlot.pyimport("matplotlib")
include("publication_plot_style.jl")

function main()
    # ===========================================================
    # USER SETTINGS — edit the default parameter choices here
    # ===========================================================
    R                 = 4.0          # Floquet parameter (τ = 2π / (ω * R))
    K                 = 1.5          # Kicking strength
    N                 = 25000        # Hilbert space dimension
    OMEGA             = 2 / π        # Oscillator frequency
    N0                = 0            # Fock / number state index (|n0⟩)
    X0                = -0.5903      # Coherent state position coordinate x0
    P0                = -8.1437      # Coherent state momentum coordinate p0
    TMIN              = 5      # Minimum time for QFI plot (e.g. 0.0 or nothing for full range)
    TMAX              = 700      # Maximum time for QFI plot (e.g. 500.0 or nothing for full range)

    OUTPUT_FILE       = "qfi_vs_time_energy.pdf"   # plot file name inside plot/

    # Left panel (QFI vs time): fixed ~t^m guide line, anchored at
    # (T_START, Y_AT_START) with exponent M_POWER.
    SHOW_LINE_M       = true
    T_START_M         = 50         # guide start t coordinate
    T_END_M           = TMAX        # guide end t coordinate
    Y_AT_START_M      = 10^5       # guide y coordinate at t = T_START_M
    M_POWER           = 5.7           # exponent of the ~t^m line

    # Right panel (mean energy): two fixed guide lines, one per curve.
    # Anchored at (T_START_E, Y_AT_START_X) with exponent X_POWER.
    SHOW_ENERGY_GUIDES = true
    T_START_E         = 100.0
    T_END_E           = TMAX
    Y_AT_START_NUM    = 15          # guide y at T_START_E for the number state
    Y_AT_START_COH    = 35.0        # guide y at T_START_E for the coherent state
    NUM_POWER         = 2            # ~t^{NUM_POWER} guide for energy_number
    COH_POWER         = 1.8           # ~t^{COH_POWER} guide for energy_coherent

    TICK_LABELSIZE    = 14           # font size of the numeric tick labels
    AXIS_LABELSIZE    = 22           # font size of the x/y axis labels
    LEGEND_FONTSIZE   = 22           # font size of the legend text
    TEXT_FONTSIZE     = 25           # font size of titles / parameter text
    CURVE_LINEWIDTH   = 3            # line width of every data curve
    SHOW_PARAM_BOX    = true         # whether to show corner parameter boxes
    # ===========================================================

    data_dir = joinpath(@__DIR__, "..", "data")
    plot_dir = joinpath(@__DIR__, "..", "plot")

    input_file  = get(ENV, "INPUT_FILE",  joinpath(data_dir, "qfi_simulations.jld2"))
    energy_file = get(ENV, "ENERGY_INPUT_FILE", joinpath(data_dir, "mean_energy_vs_time.jld2"))
    output_file = get(ENV, "OUTPUT_FILE", joinpath(plot_dir, OUTPUT_FILE))

    isfile(input_file)  || throw(ArgumentError("input file not found: $(normpath(input_file)). " *
                                               "Run scripts/calculate_qfi_time_states.jl first."))
    isfile(energy_file) || throw(ArgumentError("energy file not found: $(normpath(energy_file)). " *
                                               "Run scripts/compute_mean_energy.jl first."))

    # Target parameters (from ENV or defaults)
    target_R     = parse(Float64, get(ENV, "R", string(R)))
    target_R     = Int64(target_R)
    target_K     = parse(Float64, get(ENV, "K", string(K)))
    target_N     = parse(Int, get(ENV, "N", string(N)))
    target_omega = parse(Float64, get(ENV, "OMEGA", string(OMEGA)))
    target_n0    = parse(Int, get(ENV, "N0", string(N0)))
    target_x0    = parse(Float64, get(ENV, "X0", string(X0)))
    target_p0    = parse(Float64, get(ENV, "P0", string(P0)))
    target_tmin  = let val = get(ENV, "TMIN", "")
        isempty(strip(val)) ? TMIN : parse(Float64, val)
    end
    target_tmax  = let val = get(ENV, "TMAX", "")
        isempty(strip(val)) ? TMAX : parse(Float64, val)
    end

    println("===========================================================")
    println("Plotting QFI & Mean Energy vs Time")
    println("Target Parameters:")
    println("  R = $target_R, K = $target_K, N = $target_N, omega = $(round(target_omega, sigdigits=5))")
    println("  Fock state n0 = $target_n0")
    println("  Coherent state (x0, p0) = ($target_x0, $target_p0)")
    if !isnothing(target_tmin) || !isnothing(target_tmax)
        println("  QFI time range: [Tmin=$(target_tmin), Tmax=$(target_tmax)]")
    end
    println("===========================================================")

    # ------------------------------------------------------------------
    # Helper: Extract R for a QFI archive entry
    # ------------------------------------------------------------------
    function get_qfi_R(key::AbstractString, dict::AbstractDict)
        if haskey(dict, "R")
            return Float64(dict["R"])
        elseif haskey(dict, "omega") && haskey(dict, "tau") && dict["omega"] > 0 && dict["tau"] > 0
            return 2π / (Float64(dict["omega"]) * Float64(dict["tau"]))
        else
            parts = Base.split(key, "_")
            i = findfirst(p -> Base.startswith(p, "R"), parts)
            !isnothing(i) && return parse(Float64, Base.replace(parts[i], "R" => ""))
        end
        error("Cannot extract R for key: $key")
    end

    # ------------------------------------------------------------------
    # Load and match QFI entries (LEFT PANEL)
    # ------------------------------------------------------------------
    qfi_entries = jldopen(input_file, "r") do f
        Dict{String, Any}([k => f[k] for k in keys(f)])
    end
    isempty(qfi_entries) && throw(ArgumentError("QFI archive is empty: $(normpath(input_file))."))

    function find_qfi_entry(is_scalar::Bool)
        candidates = String[]
        for (k, d) in qfi_entries
            st = get(d, "state", nothing)
            entry_R = get_qfi_R(k, d)
            entry_K = Float64(get(d, "K", NaN))
            entry_N = Int(get(d, "N", 0))
            entry_w = Float64(get(d, "omega", NaN))

            # Match physics parameters
            params_match = isapprox(entry_R, target_R; atol=1e-4) &&
                           isapprox(entry_K, target_K; atol=1e-4) &&
                           entry_N == target_N &&
                           isapprox(entry_w, target_omega; atol=1e-4)

            !params_match && continue

            if is_scalar
                # Match Fock / number state
                if st isa Integer && st == target_n0
                    push!(candidates, k)
                elseif !Base.occursin("coh_", k)
                    push!(candidates, k)
                end
            else
                # Match Coherent state
                if (st isa Tuple || st isa AbstractVector) && length(st) >= 2
                    if isapprox(Float64(st[1]), target_x0; atol=1e-3) &&
                       isapprox(Float64(st[2]), target_p0; atol=1e-3)
                        push!(candidates, k)
                    end
                elseif Base.occursin("coh_", k)
                    push!(candidates, k)
                end
            end
        end

        if isempty(candidates)
            state_desc = is_scalar ? "Fock state |n0=$target_n0⟩" : "Coherent state (x0=$target_x0, p0=$target_p0)"
            println("\n❌ ERROR: No matching QFI entry found for $state_desc with parameters:")
            println("   R=$target_R, K=$target_K, N=$target_N, omega=$(round(target_omega, sigdigits=5))")
            println("\nAvailable QFI entries in $(normpath(input_file)):")
            for (k, d) in sort(collect(qfi_entries), by=x->x[1])
                println("  * Key: $k")
                println("    -> N=$(get(d, "N", "?")), K=$(get(d, "K", "?")), R=$(round(get_qfi_R(k, d), digits=2)), omega=$(round(get(d, "omega", NaN), sigdigits=5)), state=$(get(d, "state", "?")), t_max=$(get(d, "t_max", "?"))")
            end
            throw(ArgumentError("Missing QFI simulation run in $input_file for target parameters."))
        end

        return qfi_entries[candidates[1]]
    end

    qfi_vac = find_qfi_entry(true)
    qfi_coh = find_qfi_entry(false)

    # ------------------------------------------------------------------
    # Load and match Mean Energy dataset (RIGHT PANEL)
    # ------------------------------------------------------------------
    datasets = jldopen(energy_file, "r") do f
        Vector(f["datasets"])
    end
    isempty(datasets) && throw(ArgumentError("Energy archive holds no datasets: $(normpath(energy_file))."))

    function find_energy_dataset()
        hit = findfirst(datasets) do ds
            isapprox(Float64(ds.R), target_R; atol=1e-4) &&
            isapprox(Float64(ds.K), target_K; atol=1e-4) &&
            Int(ds.N) == target_N &&
            isapprox(Float64(ds.omega), target_omega; atol=1e-4) &&
            Int(ds.n) == target_n0 &&
            isapprox(Float64(ds.x0), target_x0; atol=1e-3) &&
            isapprox(Float64(ds.p0), target_p0; atol=1e-3)
        end

        if isnothing(hit)
            println("\n❌ ERROR: No matching Mean Energy dataset found with parameters:")
            println("   R=$target_R, K=$target_K, N=$target_N, omega=$(round(target_omega, sigdigits=5)), n0=$target_n0, (x0,p0)=($target_x0, $target_p0)")
            println("\nAvailable Mean Energy datasets in $(normpath(energy_file)):")
            for (i, ds) in enumerate(datasets)
                println("  * Dataset #$i: N=$(ds.N), K=$(ds.K), R=$(ds.R), omega=$(round(ds.omega, sigdigits=5)), n=$(ds.n), (x0,p0)=($(round(ds.x0, digits=4)), $(round(ds.p0, digits=4))), t_max=$(ds.t_max)")
            end
            throw(ArgumentError("Missing Mean Energy dataset in $energy_file for target parameters."))
        end

        return datasets[hit]
    end

    energy = find_energy_dataset()

    # ------------------------------------------------------------------
    # Time axes (physical time)
    # ------------------------------------------------------------------
    tL = qfi_vac["tau"] .* vec(0:(qfi_vac["t_max"] - 1))
    tR = energy.tau .* vec(0:(length(energy.energy_number) - 1))

    # Apply Tmin / Tmax filtering to QFI (Left Panel)
    maskL = trues(length(tL))
    if !isnothing(target_tmin)
        maskL .&= (tL .>= target_tmin)
    end
    if !isnothing(target_tmax)
        maskL .&= (tL .<= target_tmax)
    end
    count(maskL) > 0 || error("No QFI data points in time range [Tmin=$target_tmin, Tmax=$target_tmax]. Available: $(first(tL)) to $(last(tL))")

    tL_plot = tL[maskL]
    qfi_vac_plot = vec(qfi_vac["qfi"])[maskL]
    qfi_coh_plot = vec(qfi_coh["qfi"])[maskL]

    # ------------------------------------------------------------------
    # Figure setup (two side-by-side publication panels)
    # ------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5))
    apply_publication_style!(axL; scale=:loglog)
    apply_publication_style!(axR; scale=:loglog)
    mpl.rcParams["font.size"] = TICK_LABELSIZE

    state_labels = [
        target_n0 == 0 ? raw"$\vert 0\rangle$" : raw"$\vert " * string(target_n0) * raw"\rangle$",
        raw"$\vert\alpha\rangle$"
    ]

    # ------------------------------------------------------------------
    # LEFT PANEL: QFI vs time for |n0⟩ and |α⟩ (log-log)
    # ------------------------------------------------------------------
    plot_curves(tL_plot, [qfi_vac_plot, qfi_coh_plot],
                ax=axL,
                xlabel=raw"$t$",
                ylabel=raw"$\mathcal{I}(\omega; \vert \psi \rangle)$",
                labels=state_labels,
                linewidth=CURVE_LINEWIDTH,
                label_fontsize=AXIS_LABELSIZE,
                tick_fontsize=TICK_LABELSIZE+4,
                legend=false)
    axL.legend(fontsize=LEGEND_FONTSIZE, loc="lower right", frameon=false)

    # Fixed ~t^m guide line on the left panel
    if SHOW_LINE_M
        T_START_M > 0 || error("guide needs T_START_M > 0.")
        T_START_M < T_END_M || error("guide needs T_START_M < T_END_M.")
        tg = exp.(range(log(T_START_M), log(T_END_M); length=100))
        yg = Y_AT_START_M .* (tg ./ T_START_M) .^ M_POWER
        axL.plot(tg, yg; color="0.3", linestyle="--", linewidth=1.6, zorder=5)
        tlab = sqrt(T_START_M * T_END_M)
        ylab = Y_AT_START_M * (tlab / T_START_M)^M_POWER
        axL.text(tlab * 1.5, ylab * 0.2, raw"$\sim t^{" * string(M_POWER) * raw"}$";
                 ha="center", va="bottom", color="0.3", fontsize=TEXT_FONTSIZE,
                 bbox=Dict("boxstyle" => "round,pad=0.2", "facecolor" => "white",
                           "edgecolor" => "none", "alpha" => 0.0))
    end

    # ------------------------------------------------------------------
    # RIGHT PANEL: Mean energy ⟨n̂⟩ vs time for |n0⟩ and |α⟩ (log-log)
    # ------------------------------------------------------------------
    mack_enerfy_number = vec(energy.energy_number)[maskL]
    mack_enerfy_cohr = vec(energy.energy_coherent)[maskL]
    plot_curves(tL_plot, [mack_enerfy_number,mack_enerfy_cohr],
                ax=axR,
                xlabel=raw"$t$",
                ylabel=raw"$\langle \hat{n} \rangle$",
                labels=state_labels,
                linewidth=CURVE_LINEWIDTH,
                label_fontsize=AXIS_LABELSIZE,
                tick_fontsize=TICK_LABELSIZE+4,
                legend=false)
    axR.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", frameon=false)
    axR.tick_params(axis="x", labelsize=20)

    if SHOW_ENERGY_GUIDES
        # number state scaling
        y0 = Y_AT_START_NUM
        p = NUM_POWER

        tg = exp.(range(log(T_START_E), log(T_END_E); length=100))
        yg = y0 .* (tg ./ T_START_E) .^ p

        axR.plot(tg, yg; color="0.3", linestyle="--", linewidth=1.6, zorder=5 )

        tlab = sqrt(T_START_E * T_END_E)
        ylab = y0 * (tlab / T_START_E)^p

        axR.text(tlab * 1.8, ylab * 1, raw"$\sim t^{" * string(p) * raw"}$"; ha="center", va="bottom", color="0.3", fontsize=TEXT_FONTSIZE, bbox=Dict("boxstyle" => "round,pad=0.2", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.0))


        # Coherent state scaling
        y0 = Y_AT_START_COH
        p = COH_POWER

        tg = exp.(range(log(T_START_E), log(T_END_E); length=100))
        yg = y0 .* (tg ./ T_START_E) .^ p

        axR.plot(tg, yg; color="0.3", linestyle="--", linewidth=1.6, zorder=5)

        tlab = sqrt(T_START_E * T_END_E)
        ylab = y0 * (tlab / T_START_E)^p

        axR.text(tlab * 0.9, ylab * 1.5, raw"$\sim t^{" * string(p) * raw"}$"; ha="center", va="bottom", color="0.3", fontsize=TEXT_FONTSIZE, bbox=Dict("boxstyle" => "round,pad=0.2", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.0))
    end

    # Panel titles (a) and (b)
    axL.set_title(raw"(a)", fontsize=TEXT_FONTSIZE)
    axR.set_title(raw"(b)", fontsize=TEXT_FONTSIZE)

    # Optional parameter summary box
    if SHOW_PARAM_BOX
        # The box shows only the three physical parameters K, omega, and R, as
        # in scripts/plot_qfi_vs_R_at_times.jl.
        param_text = "K = $(target_K)\n\$\\omega = $(round(target_omega, sigdigits=5))\$" *
                     "\nR = $(target_R)"
        axL.text(0.3, 0.6, param_text; transform=axL.transAxes, ha="center", va="bottom",
                 fontsize=TEXT_FONTSIZE - 4, bbox=Dict("boxstyle" => "round,pad=0.3",
                 "facecolor" => "white", "edgecolor" => "0.65", "alpha" => 0.9))
        axR.text(0.75, 0.08, param_text; transform=axR.transAxes,
                 ha="center", va="bottom", fontsize=TEXT_FONTSIZE - 4,
                 bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                 "edgecolor" => "0.65", "alpha" => 0.9))
    end

    fig.tight_layout()
    mkpath(dirname(output_file))
    save_publication_figure(fig, output_file)
    PythonPlot.plotclose()

    println("✅ Saved QFI-vs-time and mean-energy-vs-time plot to: ", normpath(output_file))
    return nothing
end

main()