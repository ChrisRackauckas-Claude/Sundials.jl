using SciMLBase: alg_order
using Sundials
using Test

# These private test types exercise the generic contract without depending on
# any concrete Sundials solver implementation.
struct TestODEAlgorithm <: Sundials.SundialsODEAlgorithm{:TestMethod, :TestLinearSolver}
    max_order::Int
end

struct TestDAEAlgorithm <: Sundials.SundialsDAEAlgorithm{:TestLinearSolver}
    max_order::Int
end

@testset "Sundials algorithm interface" begin
    ode_alg = TestODEAlgorithm(7)
    dae_alg = TestDAEAlgorithm(3)

    @test alg_order(ode_alg) == 7
    @test alg_order(dae_alg) == 3
    @test Sundials.method_choice(ode_alg) === :TestMethod
    @test Sundials.method_choice(dae_alg) === :Newton
    @test Sundials.linear_solver(ode_alg) === :TestLinearSolver
    @test Sundials.linear_solver(dae_alg) === :TestLinearSolver
end
