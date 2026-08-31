# scripts/calculate_qfi_time_states.jl

using LinearAlgebra
using nonKam
using JLD2

# --------- Parameters ---------
R = 4.0          # Parameter related to the kicking period (ωτ = 2π/R)
ω = 2/pi          # Oscillator frequency
τ = 2π/(R * ω)   # Kicking period
K = 1.5        # Kicking strength
ϵ = 1e-6      # Perturbation parameter

# Simulation parameters
N = 25000        # Hilbert space dimension
t_max = 1000    # Maximum time steps for the simulation

# ----- State Type -----
# state = (0.0121 + 3*π, 0.0121) # Tuple (x0, p0) or Integer n
# state = (-0.5903, -8.1437)
state = 0
# state = 20
# --------- Setup Archive Path ---------
data_dir = joinpath(pwd(), "data")
!ispath(data_dir) && mkdir(data_dir)
archive_file = joinpath(data_dir, "qfi_simulations.jld2")

# Generate a compact dictionary key for this specific parameter set
state_id = state isa Tuple ? "coh_x$(round(state[1], digits=3))_p$(round(state[2], digits=3))" : "fock_n$(state)"
run_id = "N$(N)_K$(K)_R$(R)_w$(round(ω, digits=3))_$(state_id)"

# --------- Check Archive & Run ---------
run_simulation = true

if isfile(archive_file)
    jldopen(archive_file, "r") do file
        if haskey(file, run_id)
            println("⚠️  Found existing run: $run_id")
            print("Action - [o]verwrite or [s]kip? (o/s): ")
            if lowercase(strip(readline())) == "o"
                println("Overwriting... Old data will be cleared.")
                # We handle the deletion down below right before saving
            else
                println("Aborted. Keeping existing data.")
                global run_simulation = false
            end
        end
    end
end

if run_simulation
    println("\n🚀 Running KHO Simulation | $run_id")
    qfi = floquet_qfi(N, ω, τ, K, ϵ, t_max, state)
    
    # Store data alongside its metadata package
    data_dict = Dict(
        "qfi" => qfi, "N" => N, "omega" => ω, "tau" => τ,
        "K" => K, "epsilon" => ϵ, "t_max" => t_max, "state" => state
    )
    
    # Open in read-write mode to check and delete the duplicate key safely
    jldopen(archive_file, "a+") do file
        if haskey(file, run_id)
            delete!(file, run_id) # Explicitly wipe out the old data entry first
        end
        file[run_id] = data_dict
    end
    println("✅ Saved to archive under key: $run_id\n")
end
