"""
    mean_energy_with_time_for_U_a(U, a, adag, t_max, ψ0)

Calculates an array of mean energy values ⟨n⟩ = ⟨a^† a⟩ for each time step 
up to `t_max`, given a pure initial state `ψ0`.
"""
function mean_energy_with_time_for_U_a(U::AbstractMatrix, a::AbstractMatrix, adag::AbstractMatrix, t_max::Int, ψ0::Vector)
    # 1. Base operator (Number operator)
    n_op = adag * a
    
    # Pre-allocate array for mean energy values
    energy_array = Vector{Float64}(undef, t_max)
    
    # Initialize the current state vector
    ψ_current = copy(ψ0)
    
    # Helper function for efficient expectation value calculation
    function calc_mean_energy(n_matrix, ψ)
        # dot(ψ, n_matrix * ψ) = ⟨ψ|n|ψ⟩
        return real(dot(ψ, n_matrix * ψ))
    end
    
    # Iterative Time Evolution for t = 1 to t_max
    for j in 1:t_max
        # Schrödinger evolution: Evolve the state by one kick
        # |ψ(j)⟩ = U |ψ(j-1)⟩
        ψ_current = U * ψ_current 
        
        # Calculate and store the mean energy for the current time
        energy_array[j] = calc_mean_energy(n_op, ψ_current)
    end
    
    return energy_array
end