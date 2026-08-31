"""
    coherent_state(x0::Real, p0::Real, ω::Real, N::Int)

Generates a quantum coherent state in the truncated Fock (number) basis 
centered at classical phase space coordinates (`x0`, `p0`).

# Arguments
- `x0`: Initial expected position in phase space.
- `p0`: Initial expected momentum in phase space.
- `ω`: Angular frequency of the harmonic oscillator.
- `N`: Dimension of the truncated Hilbert space (number of energy levels).

# Returns
- `Vector{ComplexF64}`: The normalized state vector of length `N`.

# Mathematical Details
The function maps the real coordinates to a complex amplitude:
\$\\alpha = \\sqrt{\\frac{\\omega}{2}} x_0 + i \\frac{p_0}{\\sqrt{2\\omega}}\$

It then iteratively constructs the state \$\\sum e^{-|\\alpha|^2/2} \\frac{\\alpha^n}{\\sqrt{n!}} |n\\rangle\$ 
to avoid integer overflow from calculating factorials of large `N`.
"""
function coherent_state(x0::Real, p0::Real, ω::Real, N::Int)
    # 1. Convert phase space coordinates to complex amplitude α
    # Note: 'im' is the built-in imaginary unit in Julia
    α = sqrt(ω / 2) * x0 + im * (p0 / sqrt(2 * ω))
    
    # 2. Pre-allocate the complex state vector
    state = zeros(ComplexF64, N)
    
    # 3. Calculate the ground state (n = 0) amplitude
    state[1] = exp(-abs2(α) / 2) + 0.0im
    
    # 4. Iteratively calculate higher energy levels
    # This specifically avoids calculating factorial(N) which causes overflow
    for n in 1:(N - 1)
        state[n + 1] = state[n] * α / sqrt(n)
    end
    
    # 5. Re-normalize to correct for any probability mass lost to truncation
    return state / norm(state)
end