"""
    calculate_qfi(ψ_t::Vector, ψ_ϵ_t::Vector, ϵ::Real)

Calculate the Quantum Fisher Information (QFI) at a specific time step using the 
fidelity susceptibility between the unperturbed state `ψ_t` and the perturbed state `ψ_ϵ_t`.

# Arguments
- `ψ_t::Vector`: The time-evolved state vector under the unperturbed unitary operator.
- `ψ_ϵ_t::Vector`: The time-evolved state vector under the perturbed unitary operator.
- `ϵ::Real`: The perturbation parameter.

# Returns
- `Float64`: The calculated Quantum Fisher Information value.
"""
function calculate_qfi_fidility(ψ_t::Vector, ψ_ϵ_t::Vector, ϵ::Real)
    # Calculate the fidelity (quantum overlap squared) safely
    fidelity = abs(dot(ψ_t, ψ_ϵ_t))^2

    # Calculate QFI via the fidelity susceptibility formula
    qfi = (4.0 * (1.0 - fidelity)) / (ϵ^2)
    
    return qfi
end