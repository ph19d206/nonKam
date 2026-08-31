# qfi_with_time
"""
This script calculates the Quantum Fisher Information (QFI) for a kicked harmonic oscillator as a function
of time.
"""
# --------- Parameters ---------
# KHO parameters
ω = 1.0
τ = 1.0
K = 0.8

# Simulation parameters
N = 500          # Hilbert space dimension
t_max = 10      # Maximum time steps for QFI calculation

# Initial state (ground state)
ψ0 = zeros(ComplexF64, N)
ψ0[1] = 1.0 + 0im  # |0⟩ state

# --------- Main Execution ---------
# import packages
using LinearAlgebra
using nonKam

# ladder operators and Floquet unitary
a = diagm(1 => sqrt.(1:N-1))
adag = a'
U = KickedHO_floquet(N, ω, τ, K)

# Calculate QFI over time
qfi_array = qfi_with_time_provided_U_a(U, a, adag, ω, τ, K, t_max, ψ0)