# qfi_with_time_provided_U_a

"""
    qfi_with_time_provided_U_a(U, a, adag, ω, τ, K, t_max, ψ0)

Calculates an array of Quantum Fisher Information (QFI) values for each time step 
up to `t_max`, given a pure initial state `ψ0`.
"""
function qfi_with_time_provided_U_a(U::AbstractMatrix, a::AbstractMatrix, adag::AbstractMatrix, ω::Real, τ::Real, K::Real, t_max::Int, ψ0::Vector)
    # 1. Base operators
    n_op = adag * a
    
    # Ensure dense matrix for matrix functions like sin()
    A_op = Matrix((a * exp(-im * ω * τ) + adag * exp(im * ω * τ)) / sqrt(2 * ω))
    
    # 2. Single-kick generator (O_0)
    O_0 = τ * n_op + (K / (2 * ω)) * A_op * sin(A_op)
    
    # Pre-allocate array for QFI values
    qfi_array = Vector{Float64}(undef, t_max)
    
    # Initialize total operator and current operator
    O_total = copy(O_0)
    O_current = copy(O_0)
    
    # Pre-calculate adjoint of U to avoid doing it inside the loop
    U_adj = U'
    
    # Helper function for efficient variance calculation (matrix-vector ops)
    function calc_qfi(O, ψ)
        Oψ = O * ψ
        exp_val = real(dot(ψ, Oψ))       # <ψ|O|ψ>
        exp_sq = real(dot(Oψ, Oψ))       # <ψ|O^2|ψ> = ||O|ψ>||^2
        return 4.0 * (exp_sq - exp_val^2)
    end
    
    # Calculate QFI for the first kick (t = 1)
    qfi_array[1] = calc_qfi(O_total, ψ0)
    
    # Iterative Time Evolution for t = 2 to t_max
    for j in 2:t_max
        # Heisenberg evolution of the single-step operator: O(j) = U^\dagger * O(j-1) * U
        O_current = U_adj * O_current * U 
        
        # Accumulate the generator
        O_total += O_current
        
        # Calculate and store the QFI for the current total time
        qfi_array[j] = calc_qfi(O_total, ψ0)
    end
    
    return qfi_array
end