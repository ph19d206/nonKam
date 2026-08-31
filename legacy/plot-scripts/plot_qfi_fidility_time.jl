# =============================================================================
# Script: plot_qfi_combined.jl
# Description: Reads both Vacuum and Coherent QFI datasets from JLD2 and 
#              generates a combined publication-grade comparison PDF plot.
# =============================================================================

using JLD2
using PyCall

# Import Python packages for plotting exactly as specified
plt = pyimport("matplotlib.pyplot")
mpl = pyimport("matplotlib")
np = pyimport("numpy")

# 1. Configuration & Paths
data_filepath = "data/qfi_from_fidelity_with_time.jld2"
plot_dir = "plot"
mkpath(plot_dir)

# Define the physical parameters shared by both runs
K = 0.5
R = 5.0
ω = 4.0

# Coherent state coordinates to look up
# x0, p0 = -0.5903, -8.1437
x0, p0 = 0.0121 + 3*π, 0.0121
n = 0  # Vacuum state
n = round(Int, n)
# Format rounded parameters matching the keys stored by your saving function
K_r = round(K, digits=4)
R_r = round(R, digits=4)
ω_r = round(ω, digits=4)
x_val = round(x0, digits=4)
p_val = round(p0, digits=4)
# x_val = x0
# p_val = p0

# 2. Extract Data safely from JLD2 and execute Plotting
if !isfile(data_filepath)
    error("Data file missing at $data_filepath. Please run your simulation script first.")
end

# Define both keys explicitly
vacuum_key   = "number/n$(n)/K$(K_r)_R$(R_r)_W$(ω_r)"
coherent_key = "coherent/x$(x_val)_p$(p_val)/K$(K_r)_R$(R_r)_W$(ω_r)"

jldopen(data_filepath, "r") do file
    # Initialize the plot canvas
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(axis="both", labelsize=18)
    
    # Track if we successfully plotted at least one dataset
    plotted_any = false

    # --- Plot Vacuum State Data ---
    if haskey(file, vacuum_key)
        qfi_vac = file[vacuum_key]
        time_array = 1:length(qfi_vac)
        ax.plot(time_array, qfi_vac, label="\$\\left|0\\right>\$", 
            color="#db5f57",
            linewidth=2.0,
            marker="o")
        plotted_any = true
    else
        @warn "Vacuum key not found: $vacuum_key"
    end

    # --- Plot Coherent State Data ---
    if haskey(file, coherent_key)
        qfi_coh = file[coherent_key]
        time_array = 1:length(qfi_coh)
        # Using a dashed red line to cleanly differentiate it from the vacuum data
        label_coh = "\$\\left|\\alpha\\right>\$"
        ax.plot(time_array, qfi_coh, label=label_coh, 
            color="#dbdb57",
            linewidth=2.0,
            marker="o")
        plotted_any = true
    else
        @warn "Coherent key not found: $coherent_key"
    end

    # Style and save only if we found data to plot
    if plotted_any
        # Styling parameters
        ax.set_xlabel("\$t\$", fontsize=18)
        ax.set_ylabel("\$I(\\omega;\\left|\\psi\\right>)\$", fontsize=18)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(false) 
        ax.legend(fontsize=12, loc="best")
        
        # 3. Save Figure 
        pdf_filename = "$(plot_dir)/qfi_combined_K$(K_r)_R$(R_r).pdf"
        fig.savefig(pdf_filename, bbox_inches="tight")
        println("Successfully generated combined plot at: $pdf_filename")
    else
        @error "Neither Vacuum nor Coherent datasets were found in the file. Plot aborted."
    end
    
    # 4. Clean up memory
    plt.close(fig)
end