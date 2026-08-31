# eigv_kikU
"""
    eigv_kikU(N, ω, τ, K)
Constructs the N x N Floquet unitary for kicked part interms of eigenvalues and eigenvectors.
"""
function eigv_kikU(N::Int, ω::Real, K::Real)
    # 1. Load real eigenvalues and eigenvectors of X (X = (a+a^†)/sqrt(2))
    λx, V = eigv_xop(N) # λx is a Vector{Float64}
    
    # 2. Scale the position coordinates correctly for your physical basis: 
    # X_scaled = X / sqrt(ω)
    λx_scaled = λx ./ sqrt(ω) 
    
    # 3. Compute the complex Floquet eigenvalues: exp(-i * K * cos(X_scaled))
    # By assigning to a new variable name without '.=', Julia creates a Vector{ComplexF64} automatically
    λ = exp.(-im * K .* cos.(λx_scaled)) 

    return λ, V
end