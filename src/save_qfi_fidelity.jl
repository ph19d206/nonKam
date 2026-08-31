# save_qfi_fidelity
"""
    save_qfi_fidelity!(qfit::Vector{Float64}, state_type::String, K::Real, R::Real, ω::Real; x0=nothing, p0=nothing,n=nothing)

Save the calculated QFI trajectory array to a structured JLD2 file. Automatically determines
if the state is 'number' or 'coherent' based on the presence of `x0` and `p0`.

# Arguments
- `qfit::Vector{Float64}`: The array of QFI values over time
- `state_type::String`: Type of the initial state ('number', 'coherent')
- `ω::Real`: Oscillator frequency
- `R::Real`: Parameter related to the kicking period (ωτ = 2π/R)
- `K::Real`: Kicking strength
- `x0::Union{Nothing, Real}`: Initial position for coherent state (optional)
- `p0::Union{Nothing, Real}`: Initial momentum for coherent state (optional)
- `n::Union{Nothing, Int}`: Principal quantum number for number state (optional)
"""
function save_qfi_fidelity!(qfit::Vector{Float64}, state_type::String, K::Real, R::Real, ω::Real; x0=nothing, p0=nothing, n=nothing)
    # Ensure "data" directory exists
    mkpath("data")
    filepath = "data/qfi_from_fidelity_with_time.jld2"

    # Normalize input string to lowercase to avoid "Vacuum" vs "vacuum" mismatch bugs
    stype = lowercase(state_type)

    # Round parameters to 4 decimal places for clean, readable file keys
    K_r = round(K, digits=4)
    R_r = round(R, digits=4)
    ω_r = round(ω, digits=4)

    # 1. Determine the exact key path based on explicit state type
    key = if stype == "number"
        n_val = isnothing(n) ? 0 : n
        "number/n$(n_val)/K$(K_r)_R$(R_r)_W$(ω_r)"

    elseif stype == "coherent"
        # Fallback defaults if the user accidentally skipped passing them
        x_val = isnothing(x0) ? 0.0 : round(x0, digits=4)
        p_val = isnothing(p0) ? 0.0 : round(p0, digits=4)
        "coherent/x$(x_val)_p$(p_val)/K$(K_r)_R$(R_r)_W$(ω_r)"
    else
        error("Unknown state_type: '\$state_type'. Choose from 'number', 'coherent'.")
    end

    # 2. Open and safely append to the file
    jldopen(filepath, "a+") do file
        # --- Overwrite Logic ---
        # If the key already exists, delete it first to prevent the ArgumentError
        if haskey(file, key)
            delete!(file, key)
        end

        file[key] = qfit
    end
    
    println("Saved $state_type QFI to: $key")
end