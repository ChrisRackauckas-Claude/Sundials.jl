using Sundials, Test
using SciMLBase: SymbolCache

# `SymbolCache` resolves an `Expr` as an observed quantity, so `:(x + y)` is not a component
# of the state vector and cannot be produced by the dense-output interpolant.
f = ODEFunction(
    (du, u, p, t) -> (du[1] = p[1] * u[2]; du[2] = -p[1] * u[1]);
    sys = SymbolCache([:x, :y], [:ω], :t)
)
prob = ODEProblem(f, [1.0, 0.0], (0.0, 10.0), [2.0])

@testset "Symbolic idxs in the current-step interpolant: $(nameof(typeof(alg)))" for alg in
    (CVODE_Adams(), CVODE_BDF(), ARKODE())

    integrator = init(prob, alg)
    step!(integrator, 1.0, true)
    mid = (integrator.tprev + integrator.t) / 2

    # A symbolic index evaluates the whole-state interpolant and then projects, while an
    # integer index evaluates the interpolant for that component alone. The two orders of
    # operations agree to rounding rather than bitwise, so these compare with `≈`.
    @test integrator(mid; idxs = :x) ≈ integrator(mid; idxs = 1)
    @test integrator(mid; idxs = :y) ≈ integrator(mid; idxs = 2)
    @test integrator(mid; idxs = [:x, :y]) == integrator(mid)
    @test integrator(mid; idxs = :(x + y)) ≈ sum(integrator(mid))

    # Symbolic routing does not disturb the existing integer and `nothing` paths
    @test integrator(mid) == integrator(mid; idxs = nothing)
    @test integrator(mid; idxs = [1, 2]) == integrator(mid)
    @test integrator(mid; idxs = 1) == integrator(mid)[1]
end

@testset "Symbolic idxs on an IDAIntegrator" begin
    dae = DAEFunction(
        (out, du, u, p, t) -> (out[1] = du[1] - p[1] * u[2]; out[2] = du[2] + p[1] * u[1]);
        sys = SymbolCache([:x, :y], [:ω], :t)
    )
    daeprob = DAEProblem(
        dae, [0.0, -2.0], [1.0, 0.0], (0.0, 10.0), [2.0];
        differential_vars = [true, true]
    )
    integrator = init(daeprob, IDA())
    step!(integrator, 1.0, true)
    mid = (integrator.tprev + integrator.t) / 2

    @test integrator(mid; idxs = :x) ≈ integrator(mid; idxs = 1)
    @test integrator(mid; idxs = :(x + y)) ≈ sum(integrator(mid))
    @test integrator(mid; idxs = [1, 2]) == integrator(mid)
end
