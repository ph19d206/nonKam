using nonKam
using LinearAlgebra
using JLD2
using Test

@testset "nonKam.jl Total Suite" begin

    @testset "Coherent States" begin
        # This executes the tests inside test_coherent_state.jl
        include("test_coherent_state.jl")
    end

    @testset "Quadrature distribution" begin
        # This executes the tests inside test_quadrature.jl
        include("test_quadrature.jl")
    end

    @testset "Number-operator variance" begin
        # A number state is an eigenstate of n̂, so its variance at t = 0 is zero.
        @test var_n(8, 1.0, 1.0, 0.5, 1, 3)[1] ≈ 0.0 atol=1e-14

        # For a coherent state, Var(n̂) equals ⟨n̂⟩ (up to finite-basis truncation).
        coherent_mean = mean_n(80, 1.0, 1.0, 0.0, 1, (1.0, 0.0))[1]
        coherent_variance = var_n(80, 1.0, 1.0, 0.0, 1, (1.0, 0.0))[1]
        @test coherent_variance ≈ coherent_mean atol=1e-12
    end

    @testset "Intermediate power-law exponent" begin
        t = collect(1.0:300.0)
        # Short-time transient (m = 0.5), physical scaling window (m = 2),
        # then a finite-basis saturation plateau (m = 0).
        f30 = 30.0^0.5
        f200 = f30 * (200.0 / 30.0)^2
        f = [x <= 30 ? x^0.5 : x <= 200 ? f30 * (x / 30)^2 : f200 for x in t]

        result = find_converged_m(t, f; transient_fraction=0.1,
                                  min_points=12, tol=0.02)
        @test result.m_converged ≈ 2.0 atol=1e-10
        @test first(result.t_converged) >= 30
        @test last(result.t_converged) <= 200

        regimes = find_power_law_regimes(t, f)
        @test length(regimes.regimes) == 3
        @test regimes.regimes[1].m ≈ 0.5 atol=1e-10
        @test regimes.regimes[2].m ≈ 2.0 atol=1e-10
        @test regimes.regimes[3].m ≈ 0.0 atol=1e-10
    end

    @testset "QFI R sweep" begin
        R = [1.95, 2.0, 2.05]
        sweep = floquet_qfi_R_sweep(8, 1.0, R, 0.1, 1e-5, 2, 0; parallel=false)
        @test sweep.R == R
        @test sweep.tau ≈ 2π ./ R
        @test size(sweep.qfi) == (2, 3)
        @test sweep.times[:, 2] ≈ sweep.time_steps .* sweep.tau[2]

        output_file = joinpath(tempdir(), "nonKam_qfi_R_sweep_test.jld2")
        save_qfi_R_sweep(output_file, sweep, 8, 1.0, 0.1, 1e-5, 2, 0)
        loaded = JLD2.load(output_file)
        @test loaded["R"] == R
        @test loaded["qfi"] == sweep.qfi
        @test loaded["parameters"].N == 8
    end

end
