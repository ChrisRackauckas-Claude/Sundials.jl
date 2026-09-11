using Sundials, BenchmarkTools

const SUITE = BenchmarkGroup()

# =============================================================================
# CVODE solves
# =============================================================================

SUITE["cvode"] = BenchmarkGroup()

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
    return nothing
end
ode_prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 30.0))

SUITE["cvode"]["bdf"] = @benchmarkable solve($ode_prob, CVODE_BDF())
SUITE["cvode"]["adams"] = @benchmarkable solve($ode_prob, CVODE_Adams())
SUITE["cvode"]["bdf_densejac"] = @benchmarkable solve(
    $ode_prob, CVODE_BDF(; linear_solver = :Dense)
)

# =============================================================================
# ARKODE
# =============================================================================

SUITE["arkode"] = BenchmarkGroup()

SUITE["arkode"]["explicit"] = @benchmarkable solve(
    $ode_prob, ARKODE(Sundials.Explicit())
)
SUITE["arkode"]["implicit"] = @benchmarkable solve(
    $ode_prob, ARKODE(Sundials.Implicit())
)

# =============================================================================
# IDA (DAE)
# =============================================================================

SUITE["ida"] = BenchmarkGroup()

function dae_robertson!(resid, du, u, p, t)
    resid[1] = du[1] + 0.04 * u[1] - 1.0e4 * u[2] * u[3]
    resid[2] = du[2] - 0.04 * u[1] + 1.0e4 * u[2] * u[3] + 3.0e7 * u[2]^2
    resid[3] = u[1] + u[2] + u[3] - 1.0
    return nothing
end
u0_dae = [1.0, 0.0, 0.0]
du0_dae = [-0.04, 0.04, 0.0]
dae_prob = DAEProblem(
    dae_robertson!, du0_dae, u0_dae, (0.0, 10.0);
    differential_vars = [true, true, false]
)

SUITE["ida"]["robertson"] = @benchmarkable solve($dae_prob, IDA())
