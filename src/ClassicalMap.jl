"""
    ClassicalMap(ω, ϵ, τ, u0, v0, t)

Simulates the classical map of a kicked harmonic oscillator for `t` iterations.

# Arguments
- `ω`: Angular frequency.
- `ϵ`: Kick strength.
- `τ`: Time interval between kicks.
- `u0`, `v0`: Initial position and velocity (coordinates).
- `t`: Number of time steps (integer).

# Returns
- `u_array, v_array`: Vectors containing the trajectory of the system.
"""
function ClassicalMap(ω, ϵ, τ, u0, v0, t::Int)
    u_array = Vector{Float64}(undef, t)
    v_array = Vector{Float64}(undef, t)
    
    ui, vi = u0, v0
    cos_τω = cos(τ * ω)
    sin_τω = sin(τ * ω)

    for i in 1:t
        u_array[i] = ui
        v_array[i] = vi

        # Map update logic
        # Note: ui is updated first and then used to calculate vi
        kick = ui + ϵ * sin(vi)
        ui = kick * cos_τω + vi * sin_τω
        vi = -kick * sin_τω + vi * cos_τω
    end

    return u_array, v_array
end