using JLD2
using PyCall

# Import Python packages for plotting exactly as specified
plt = pyimport("matplotlib.pyplot")
mpl = pyimport("matplotlib")
np = pyimport("numpy")

# --------- Physical Parameters ---------
R = 2.0
ω = 2/pi 
K = 0.5

# State parameter specifications
x0, p0 = 0.0121 + 3*π, 0.0121
n = 0  # Vacuum state index
N = 20000  # Hilbert space dimension

# Format rounded parameters matching the archive key configurations
K_r = round(K, digits=3)
R_r = round(R, digits=3)
ω_r = round(ω, digits=3)
x_val = round(x0, digits=3)
p_val = round(p0, digits=3)

# 1. Configuration & Paths
data_filepath = "data/qfi_simulations.jld2"  # Match your new single master file path
plot_dir = "plot"
mkpath(plot_dir)

if !isfile(data_filepath)
    error("Archive file missing at $data_filepath. Please compile data metrics first.")
end

# 2. Reconstruct your specific structural archive keys
number_key   = "N$(N)_K$(K)_R$(R)_w$(round(ω, digits=3))_fock_n$(n)"
coherent_key = "N$(N)_K$(K)_R$(R)_w$(round(ω, digits=3))_coh_x$(x_val)_p$(p_val)"

jldopen(data_filepath, "r") do file
    # Initialize the plot canvas with explicit aesthetic sizing 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(axis="both", labelsize=18) 
    
    plotted_any = false

    # --- Plot Vacuum State Data ---
    if haskey(file, number_key)
        # Extract the vector inside the stored data dictionary
        qfi_vac = file[number_key]["qfi"]
        time_array = 1:length(qfi_vac)
        
        ax.plot(time_array, qfi_vac, 
                label="\$\\ \\left|n\\right>\$", 
                color="#db5f57",   # Soft Crimson
                linewidth=2.0,
                linestyle="-",     # Solid Line
                marker="s",        # Square Markers
                markersize=4,
                markevery=max(1, length(time_array) ÷ 25) # Prevents marker crowding on large grids
                ) 
        plotted_any = true
    else
        @warn "Vacuum key missing from archive: $number_key"
    end

    # --- Plot Coherent State Data ---
    if haskey(file, coherent_key)
        # Extract the vector inside the stored data dictionary
        qfi_coh = file[coherent_key]["qfi"]
        time_array = 1:length(qfi_coh)
        
        ax.plot(time_array, qfi_coh, 
                label="\$\\ \\left|\\alpha\\right>\$", 
                color="#4e79a7",   # Deep Slate Blue (Replaced muddy yellow for better contrast)
                linewidth=2.0,
                linestyle="-",    # Solid Line
                marker="o",        # Circle Markers
                markersize=4,
                markevery=max(1, length(time_array) ÷ 25)
                )
        plotted_any = true
    else
        @warn "Coherent key missing from archive: $coherent_key"
    end

    # Style and save only if data exists
    if plotted_any
        ax.plot(time_array[100:end], (0.11*time_array[100:end]).^6, 
                color="black",
                linewidth=2.0,
                linestyle="--",   
                markevery=max(1, length(time_array) ÷ 25)
            )
       
        param_text = "\$\\ \\sim t^6\$"

        ax.text(0.82, 0.65, param_text,
        color="black", 
        transform=ax.transAxes,
        fontsize=16
        )

        ax.plot(time_array[1:40], (12000*time_array[1:40]).^1, 
                color="red",
                linewidth=2.0,
                linestyle="--",   
                markevery=max(1, length(time_array) ÷ 25)
            )
       
        param_text = "\$\\ \\sim t^{1}\$"

        ax.text(0.2, 0.2, param_text,
        color="red",
        transform=ax.transAxes,
        fontsize=16
        )




        # Labels and Mathematical Expressions
        ax.set_xlabel("\$t\$", fontsize=18) 
        ax.set_ylabel("\$I(\\omega;\\left|\\psi\\right>)\$", fontsize=18) 
        
        # Scaling parameters
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Clean background grid framework for log scales
        # ax.grid(true, which="both", linestyle=":", alpha=0.5, color="gray")
        ax.grid(false) 

        # Legend styling
        ax.legend(fontsize=14, loc="best", frameon=true, facecolor="white")
        
        # Info Box showing static parameters in space
        param_text = "\$K = $K_r\$\n\$R = $R_r\$\n\$\\omega = $ω_r\$"
        ax.text(0.8, 0.2, param_text, 
                transform=ax.transAxes, 
                fontsize=16, 
                verticalalignment="top", 
                bbox=Dict("boxstyle"=>"round,pad=0.5", "facecolor"=>"#f5f5f5", "alpha"=>0.7, "edgecolor"=>"none"))
        
        # 3. Save Figure 
        pdf_filename = "$(plot_dir)/qfi_combined_N$(N)_K$(K_r)_R$(R_r).pdf"
        fig.savefig(pdf_filename, bbox_inches="tight")
        println("✅ Combined plot successfully generated at: $pdf_filename")
    else
        @error "Requested keys were missing. Plot generation aborted."
    end
    
    # 4. Clean up window framework memory
    plt.close(fig)
end