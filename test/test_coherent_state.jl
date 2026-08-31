# Note: We don't need to "using nonKam" here if it is 
# already called in runtests.jl

@testset "Quantum States" begin

    # =========================================================================
    # 1. Tests for coherent_state(x0, p0, ω, N)
    # =========================================================================
    @testset "coherent_state" begin
        ω = 1.0
        N = 35  # Dimension high enough to avoid truncation artifacts for small α
        
        @testset "Vacuum State Base Case" begin
            # When x0=0, p0=0 -> α=0, which is the ground fock state |0>
            vac = coherent_state(0.0, 0.0, ω, N)
            
            @test length(vac) == N
            @test norm(vac) ≈ 1.0
            @test vac[1] ≈ 1.0 + 0.0im
            # All other elements should be essentially zero
            @test all(abs2.(vac[2:end]) .< 1e-15) 
        end

        @testset "Mathematical Invariants" begin
            # Test with arbitrary physical parameters
            x0, p0 = 0.7, -0.3
            psi = coherent_state(x0, p0, ω, N)
            
            # State must always be normalized
            @test norm(psi) ≈ 1.0
            
            # Complex amplitude calculation verification: 
            # α = sqrt(ω/2)*x0 + i*(p0/sqrt(2ω))
            α = sqrt(ω / 2) * x0 + im * (p0 / sqrt(2 * ω))
            
            # The analytical inner product of |α> and |0> is e^(-|α|^2 / 2)
            # psi[1] is exactly <0|psi>
            @test psi[1] ≈ exp(-abs2(α) / 2) atol=1e-7
        end

        @testset "Physical Expectation Values" begin
            x0, p0 = 1.2, 0.4
            psi = coherent_state(x0, p0, ω, N)
            
            # Building the annihilation operator (a) in the Fock basis
            a = zeros(ComplexF64, N, N)
            for n in 1:(N-1)
                a[n, n+1] = sqrt(n)
            end
            a_dag = a'  # Use standard Julia shorthand for conjugate transpose
            
            # Define Position (X) and Momentum (P) operators
            X_op = sqrt(1 / (2 * ω)) * (a + a_dag)
            P_op = -im * sqrt(ω / 2) * (a - a_dag)
            
            # Expectation values <psi|O|psi> using explicit LinearAlgebra pathing
            # This ensures no matter how runtests.jl handles namespacing, it won't crash
            expt_x = real(LinearAlgebra.dot(psi, X_op * psi))
            expt_p = real(LinearAlgebra.dot(psi, P_op * psi))
            
            # Verify they yield the initial classical coordinates
            @test expt_x ≈ x0 atol=1e-5
            @test expt_p ≈ p0 atol=1e-5
        end
    end

    # =========================================================================
    # 2. Example placeholder for your next function (e.g., fock_state)
    # =========================================================================
    @testset "fock_state" begin
        # Placeholder for future expansion
    end

end