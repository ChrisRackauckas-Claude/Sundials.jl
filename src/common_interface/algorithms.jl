# Sundials.jl algorithms

# Abstract Types
"""
    SundialsODEAlgorithm{Method, LinearSolver}

Abstract supertype for Sundials-backed ODE algorithm values.

# Arguments
- `Method`: Sundials nonlinear iteration choice encoded in the algorithm type, such as
  `:Newton` or `:Functional`.
- `LinearSolver`: Sundials linear solver choice encoded in the algorithm type, such as
  `:Dense`, `:Band`, `:GMRES`, or `:KLU`.

# Returns
Concrete subtypes, such as [`CVODE_BDF`](@ref), [`CVODE_Adams`](@ref), and
[`ARKODE`](@ref), are algorithm objects passed to `solve`.

# Interface Notes
This type is part of the SciML common solve interface. Downstream extensions should dispatch
on concrete algorithm types when they need solver-specific options, and on
`SundialsODEAlgorithm` only for behavior shared by all Sundials ODE algorithms.
The generic Sundials methods use the type parameters for `method_choice` and
`linear_solver`, and read the `max_order` field for `SciMLBase.alg_order`.
Concrete ODE algorithms must therefore provide that field and the two type
parameters; they may add solver-specific fields and methods.

# Examples
```julia
alg = CVODE_BDF()
alg isa SundialsODEAlgorithm
```
"""
abstract type SundialsODEAlgorithm{Method, LinearSolver} <: SciMLBase.AbstractODEAlgorithm end

"""
    SundialsDAEAlgorithm{LinearSolver}

Abstract supertype for Sundials-backed DAE algorithm values.

# Arguments
- `LinearSolver`: Sundials linear solver choice encoded in the algorithm type, such as
  `:Dense`, `:Band`, `:GMRES`, or `:KLU`.

# Returns
Concrete subtypes, such as [`IDA`](@ref), are algorithm objects passed to `solve`.

# Interface Notes
This type is part of the SciML common solve interface. Downstream extensions should dispatch
on concrete algorithm types when they need solver-specific options, and on
`SundialsDAEAlgorithm` only for behavior shared by all Sundials DAE algorithms.
The generic Sundials methods use the type parameter for `linear_solver`, return
`:Newton` from `method_choice`, and read the `max_order` field for
`SciMLBase.alg_order`. Concrete DAE algorithms must provide that field and the
type parameter; they may add solver-specific fields and methods.

# Examples
```julia
alg = IDA()
alg isa SundialsDAEAlgorithm
```
"""
abstract type SundialsDAEAlgorithm{LinearSolver} <: SciMLBase.AbstractDAEAlgorithm end

"""
    SundialsNonlinearSolveAlgorithm{LinearSolver}

Developer-only implementation type for Sundials algorithms used by
NonlinearSolve. It is intentionally not exported and is not part of the user
solver API. Package developers extending KINSOL dispatch may use it together
with the internal `linear_solver` helper.
"""
abstract type SundialsNonlinearSolveAlgorithm{LinearSolver} <:
SciMLBase.AbstractNonlinearAlgorithm end

SciMLBase.alg_order(alg::Union{SundialsODEAlgorithm, SundialsDAEAlgorithm}) = alg.max_order

# ODE Algorithms
"""
    CVODE_BDF(; method = :Newton, linear_solver = :Dense, jac_upper = 0,
        jac_lower = 0, non_zero = 0, krylov_dim = 0,
        stability_limit_detect = false, max_hnil_warns = 10, max_order = 5,
        max_error_test_failures = 7, max_nonlinear_iters = 3,
        max_convergence_failures = 10, prec = nothing, psetup = nothing,
        prec_side = 0)

CVODE_BDF is CVODE's implicit backward differentiation formula method for
ordinary differential equations. Pass the resulting algorithm object to
`solve` or `init`.

# Keyword Arguments

- `method`: nonlinear iteration method, stored in the algorithm type. The
  default is `:Newton`; `:Functional` is also supported.
- `linear_solver`: linear solver backend. Supported choices include `:None`,
  `:Diagonal`, `:Dense`, `:LapackDense`, `:Band`, `:LapackBand`, `:BCG`,
  `:GMRES`, `:FGMRES`, `:PCG`, `:TFQMR`, and `:KLU`.
- `jac_upper`, `jac_lower`: upper and lower half-bandwidths. Both must be
  nonzero when `linear_solver = :Band`; they should also be set for
  `:LapackBand`.
- `non_zero`: accepted for compatibility with older Sundials.jl constructors;
  it is not stored in the algorithm object.
- `krylov_dim`: maximum Krylov subspace dimension for iterative linear solvers.
- `stability_limit_detect`: whether CVODE should detect a BDF stability limit.
- `max_hnil_warns`: maximum number of warnings for steps with negligible time
  advance.
- `max_order`: maximum BDF order.
- `max_error_test_failures`: maximum error-test failures per step.
- `max_nonlinear_iters`: maximum nonlinear iterations per step.
- `max_convergence_failures`: maximum nonlinear convergence failures.
- `prec`: preconditioner function for iterative linear solvers, or `nothing`.
- `psetup`: optional preconditioner setup function, or `nothing`.
- `prec_side`: preconditioning side passed to CVODE.

# Fields

The fields `jac_upper`, `jac_lower`, `krylov_dim`, `stability_limit_detect`,
`max_hnil_warns`, `max_order`, `max_error_test_failures`,
`max_nonlinear_iters`, `max_convergence_failures`, `prec`, `psetup`, and
`prec_side` store the corresponding constructor options. `method` and
`linear_solver` are type parameters and can be queried through the internal
solver dispatch used by Sundials.

# Returns

A `CVODE_BDF` algorithm object.

# Throws

An error is thrown when an unsupported `linear_solver` is selected or when
`:Band` is selected without both band-widths.

# Examples

```julia
prob = ODEProblem((du, u, p, t) -> (du[1] = -u[1]), [1.0], (0.0, 1.0))
solve(prob, CVODE_BDF())
CVODE_BDF(method = :Functional)
CVODE_BDF(linear_solver = :Band, jac_upper = 3, jac_lower = 3)
```
"""
struct CVODE_BDF{Method, LinearSolver, P, PS} <: SundialsODEAlgorithm{Method, LinearSolver}
    jac_upper::Int
    jac_lower::Int
    krylov_dim::Int
    stability_limit_detect::Bool
    max_hnil_warns::Int
    max_order::Int
    max_error_test_failures::Int
    max_nonlinear_iters::Int
    max_convergence_failures::Int
    prec::P
    psetup::PS
    prec_side::Int
end
function CVODE_BDF(;
        method = :Newton,
        linear_solver = :Dense,
        jac_upper = 0,
        jac_lower = 0,
        non_zero = 0,
        krylov_dim = 0,
        stability_limit_detect = false,
        max_hnil_warns = 10,
        max_order = 5,
        max_error_test_failures = 7,
        max_nonlinear_iters = 3,
        max_convergence_failures = 10,
        prec = nothing,
        psetup = nothing,
        prec_side = 0
    )
    if linear_solver == :Band && (jac_upper == 0 || jac_lower == 0)
        error("Banded solver must set the jac_upper and jac_lower")
    end
    if !(
            linear_solver in (
                :None,
                :Diagonal,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Linear solver choice not accepted.")
    end
    return CVODE_BDF{method, linear_solver, typeof(prec), typeof(psetup)}(
        jac_upper,
        jac_lower,
        krylov_dim,
        stability_limit_detect,
        max_hnil_warns,
        max_order,
        max_error_test_failures,
        max_nonlinear_iters,
        max_convergence_failures,
        prec,
        psetup,
        prec_side
    )
end
"""
    CVODE_Adams(; method = :Functional, linear_solver = :None, jac_upper = 0,
        jac_lower = 0, krylov_dim = 0, stability_limit_detect = false,
        max_hnil_warns = 10, max_order = 12, max_error_test_failures = 7,
        max_nonlinear_iters = 3, max_convergence_failures = 10,
        prec = nothing, psetup = nothing, prec_side = 0)

CVODE_Adams is CVODE's Adams-Moulton method for ordinary differential
equations. Pass the resulting algorithm object to `solve` or `init`.

# Keyword Arguments

- `method`: nonlinear iteration method, stored in the algorithm type. The
  default is `:Functional`; `:Newton` is also supported.
- `linear_solver`: linear solver backend. The default is `:None`; the same
  choices as [`CVODE_BDF`](@ref) are supported.
- `jac_upper`, `jac_lower`: upper and lower half-bandwidths. Both must be
  nonzero when `linear_solver = :Band`; they should also be set for
  `:LapackBand`.
- `krylov_dim`, `stability_limit_detect`, `max_hnil_warns`, `max_order`,
  `max_error_test_failures`, `max_nonlinear_iters`, and
  `max_convergence_failures`: corresponding CVODE iteration and order limits.
- `prec`, `psetup`, `prec_side`: iterative linear-solver preconditioner hooks.

# Fields

The fields `jac_upper`, `jac_lower`, `krylov_dim`, `stability_limit_detect`,
`max_hnil_warns`, `max_order`, `max_error_test_failures`,
`max_nonlinear_iters`, `max_convergence_failures`, `prec`, `psetup`, and
`prec_side` store the corresponding constructor options. `method` and
`linear_solver` are type parameters.

# Returns

A `CVODE_Adams` algorithm object.

# Throws

An error is thrown when an unsupported `linear_solver` is selected or when
`:Band` is selected without both band-widths.

# Examples

```julia
prob = ODEProblem((du, u, p, t) -> (du[1] = -u[1]), [1.0], (0.0, 1.0))
solve(prob, CVODE_Adams())
CVODE_Adams(method = :Newton, linear_solver = :Dense)
CVODE_Adams(linear_solver = :Band, jac_upper = 3, jac_lower = 3)
```
"""
struct CVODE_Adams{Method, LinearSolver, P, PS} <:
    SundialsODEAlgorithm{Method, LinearSolver}
    jac_upper::Int
    jac_lower::Int
    krylov_dim::Int
    stability_limit_detect::Bool
    max_hnil_warns::Int
    max_order::Int
    max_error_test_failures::Int
    max_nonlinear_iters::Int
    max_convergence_failures::Int
    prec::P
    psetup::PS
    prec_side::Int
end
function CVODE_Adams(;
        method = :Functional,
        linear_solver = :None,
        jac_upper = 0,
        jac_lower = 0,
        krylov_dim = 0,
        stability_limit_detect = false,
        max_hnil_warns = 10,
        max_order = 12,
        max_error_test_failures = 7,
        max_nonlinear_iters = 3,
        max_convergence_failures = 10,
        prec = nothing,
        psetup = nothing,
        prec_side = 0
    )
    if linear_solver == :Band && (jac_upper == 0 || jac_lower == 0)
        error("Banded solver must set the jac_upper and jac_lower")
    end
    if !(
            linear_solver in (
                :None,
                :Diagonal,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Linear solver choice not accepted.")
    end
    return CVODE_Adams{method, linear_solver, typeof(prec), typeof(psetup)}(
        jac_upper,
        jac_lower,
        krylov_dim,
        stability_limit_detect,
        max_hnil_warns,
        max_order,
        max_error_test_failures,
        max_nonlinear_iters,
        max_convergence_failures,
        prec,
        psetup,
        prec_side
    )
end
"""
    ARKODE(stiffness = Implicit(); method = :Newton, linear_solver = :Dense,
        mass_linear_solver = :Dense, jac_upper = 0, jac_lower = 0,
        mass_upper = 0, mass_lower = 0, non_zero = 0, krylov_dim = 0,
        mass_krylov_dim = 0, max_hnil_warns = 10,
        max_error_test_failures = 7, max_nonlinear_iters = 3,
        max_convergence_failures = 10, predictor_method = 0,
        nonlinear_convergence_coefficient = 0.1, dense_order = 3, order = 4,
        set_optimal_params = false, crdown = 0.3, dgmax = 0.2, rdiv = 2.3,
        msbp = 20, adaptivity_method = 0, itable = nothing, etable = nothing,
        prec = nothing, psetup = nothing, prec_side = 0)

ARKODE: Explicit and ESDIRK Runge-Kutta methods of orders 2-8 depending on choice of options.

# Arguments

- `stiffness`: selects the `Implicit()` or `Explicit()` ARK stepper.

# Keyword Arguments

- `method`: nonlinear iteration method for implicit problems.
- `linear_solver`: linear solver for the ODE part.
- `mass_linear_solver`: linear solver for a non-identity mass matrix.
- `jac_upper`, `jac_lower`: Jacobian half-bandwidths for banded solvers.
- `mass_upper`, `mass_lower`: mass-matrix half-bandwidths for banded solvers.
- `non_zero`: accepted for compatibility with older constructors; it is not
  stored in the algorithm object.
- `krylov_dim`, `mass_krylov_dim`: Krylov dimensions for the ODE and mass
  matrix linear solves.
- `max_hnil_warns`, `max_error_test_failures`, `max_nonlinear_iters`, and
  `max_convergence_failures`: iteration and error limits.
- `predictor_method`, `nonlinear_convergence_coefficient`, `dense_order`,
  `order`, `set_optimal_params`, `crdown`, `dgmax`, `rdiv`, `msbp`, and
  `adaptivity_method`: ARKODE nonlinear, interpolation, and adaptivity
  controls. `adaptivity_method` is accepted for compatibility but is not
  stored in the algorithm object.
- `itable`, `etable`: optional implicit and explicit ARK tableaux.
- `prec`, `psetup`, `prec_side`: iterative linear-solver preconditioner hooks.

# Fields

The fields `stiffness`, `jac_upper`, `jac_lower`, `mass_upper`, `mass_lower`,
`krylov_dim`, `mass_krylov_dim`, `max_hnil_warns`, `max_error_test_failures`,
`max_nonlinear_iters`, `max_convergence_failures`, `predictor_method`,
`nonlinear_convergence_coefficient`, `dense_order`, `order`,
`set_optimal_params`, `crdown`, `dgmax`, `rdiv`, `msbp`, `itable`, `etable`,
`prec`, `psetup`, and `prec_side` store the corresponding constructor
options. `method`, `linear_solver`, and `mass_linear_solver` are type
parameters.

# Returns

An `ARKODE` algorithm object for use with `SciMLBase.solve` on ODE problems.

# Throws

An error is thrown when an unsupported ODE or mass-matrix `linear_solver` is
selected, or when `:Band` is selected without both band-widths.

# Examples

## Tableau Choices

The main options for ARKODE are the choice between explicit and implicit and the method
order, given via:

```julia
ARKODE(Sundials.Explicit()) # Solve with explicit tableau of default order 4
ARKODE(Sundials.Implicit(), order = 3) # Solve with explicit tableau of order 3
```

The order choices for explicit are 2 through 8 and for implicit 3 through 5. Specific
methods can also be set through the etable and itable options for explicit and implicit
tableaus respectively. The available tableaus are:

`etable`:

* `HEUN_EULER_2_1_2`: 2nd order Heun's method
* `BOGACKI_SHAMPINE_4_2_3`: third-order method of Bogacki and Shampine
* `ARK324L2SA_ERK_4_2_3`: explicit portion of Kennedy and Carpenter's 3rd order method
* `ZONNEVELD_5_3_4`: 4th order explicit method
* `ARK436L2SA_ERK_6_3_4`: explicit portion of Kennedy and Carpenter's 4th order method
* `SAYFY_ABURUB_6_3_4`: 4th order explicit method
* `CASH_KARP_6_4_5`: 5th order explicit method
* `FEHLBERG_6_4_5`: Fehlberg's classic 5th order method
* `DORMAND_PRINCE_7_4_5`: the classic 5th order Dormand-Prince method
* `ARK548L2SA_ERK_8_4_5`: explicit portion of Kennedy and Carpenter's 5th order method
* `VERNER_8_5_6`: Verner's classic 5th order method
* `FEHLBERG_13_7_8`: Fehlberg's 8th order method

`itable`:

* `SDIRK_2_1_2`: An A-B-stable 2nd order SDIRK method
* `BILLINGTON_3_3_2`: A second order method with a 3rd order error predictor of less stability
* `TRBDF2_3_3_2`: The classic TR-BDF2 method
* `KVAERNO_4_2_3`: an L-stable 3rd order ESDIRK method
* `ARK324L2SA_DIRK_4_2_3`: implicit portion of Kennedy and Carpenter's 3th order method
* `CASH_5_2_4`: Cash's 4th order L-stable SDIRK method
* `CASH_5_3_4`: Cash's 2nd 4th order L-stable SDIRK method
* `SDIRK_5_3_4`: Hairer's 4th order SDIRK method
* `KVAERNO_5_3_4`: Kvaerno's 4th order ESDIRK method
* `ARK436L2SA_DIRK_6_3_4`: implicit portion of Kennedy and Carpenter's 4th order method
* `KVAERNO_7_4_5`: Kvaerno's 5th order ESDIRK method
* `ARK548L2SA_DIRK_8_4_5`: implicit portion of Kennedy and Carpenter's 5th order method

These can be set for example via:

```julia
ARKODE(Sundials.Explicit(), etable = Sundials.DORMAND_PRINCE_7_4_5)
ARKODE(Sundials.Implicit(), itable = Sundials.KVAERNO_4_2_3)
```

## Method Choices

* `method` - The nonlinear iteration method for implicit ARKODE problems.
* `linear_solver` - The linear solver used by implicit ARKODE problems.

## Linear Solver Choices

The choices for the linear solver are:

* `:Dense` - A dense linear solver.
* `:Band` - A solver specialized for banded Jacobians. If used, you must set the position of the upper and lower non-zero diagonals via jac_upper and jac_lower.
* `:LapackDense` - A version of the dense linear solver that uses the Julia-provided OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than :Dense on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
* `:LapackBand` - A version of the banded linear solver that uses the Julia-provided OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than :Band on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
* `:Diagonal` - This method is specialized for diagonal Jacobians.
* `:GMRES` - A GMRES method. Recommended first choice Krylov method
* `:BCG` - A Biconjugate gradient method.
* `:PCG` - A preconditioned conjugate gradient method. Only for symmetric linear systems.
* `:TFQMR` - A TFQMR method.
* `:KLU` - A sparse factorization method. Requires that the user specifies a Jacobian. The Jacobian must be set as a sparse matrix in the ODEProblem type.

### Preconditioners

Note that here `prec` is a preconditioner function
`prec(z, r, p, t, y, fy, gamma, delta, lr)` where:

- `z`: the computed output vector
- `r`: the right-hand side vector of the linear system
- `p`: the parameters
- `t`: the current independent variable
- `du`: the current value of `f(u,p,t)`
- `gamma`: the `gamma` of `W = M - gamma * J`
- `delta`: the iterative method tolerance
- `lr`: a flag for whether `lr = 1` (left) or `lr = 2` (right)
  preconditioning

and `psetup` is the preconditioner setup function for pre-computing Jacobian
information `psetup(p, t, u, du, jok, jcurPtr, gamma)`. Where:

- `p`: the parameters
- `t`: the current independent variable
- `u`: the current state
- `du`: the current `f(u,p,t)`
- `jok`: a bool indicating whether the Jacobian needs to be updated
- `jcurPtr`: a reference to an Int for whether the Jacobian was updated.
  `jcurPtr[] = true` should be set if the Jacobian was updated, and
  `jcurPtr[] = false` should be set if the Jacobian was not updated.
- `gamma`: the `gamma` of `W = M - gamma*J`

`psetup` is optional when `prec` is set.

### Additional Options

See the [ARKODE manual](https://computing.llnl.gov/sites/default/files/ark_guide-4.7.0.pdf)
for details on the additional options.
"""
struct ARKODE{Method, LinearSolver, MassLinearSolver, T, T1, T2, P, PS} <:
    SundialsODEAlgorithm{Method, LinearSolver}
    stiffness::T
    jac_upper::Int
    jac_lower::Int
    mass_upper::Int
    mass_lower::Int
    krylov_dim::Int
    mass_krylov_dim::Int
    max_hnil_warns::Int
    max_error_test_failures::Int
    max_nonlinear_iters::Int
    max_convergence_failures::Int
    predictor_method::Int
    nonlinear_convergence_coefficient::Float64
    dense_order::Int
    order::Int
    set_optimal_params::Bool
    crdown::Float64
    dgmax::Float64
    rdiv::Float64
    msbp::Int
    itable::T1
    etable::T2
    prec::P
    psetup::PS
    prec_side::Int
end

function ARKODE(
        stiffness = Implicit();
        method = :Newton,
        linear_solver = :Dense,
        mass_linear_solver = :Dense,
        jac_upper = 0,
        jac_lower = 0,
        mass_upper = 0,
        mass_lower = 0,
        non_zero = 0,
        krylov_dim = 0,
        mass_krylov_dim = 0,
        max_hnil_warns = 10,
        max_error_test_failures = 7,
        max_nonlinear_iters = 3,
        max_convergence_failures = 10,
        predictor_method = 0,
        nonlinear_convergence_coefficient = 0.1,
        dense_order = 3,
        order = 4,
        set_optimal_params = false,
        crdown = 0.3,
        dgmax = 0.2,
        rdiv = 2.3,
        msbp = 20,
        adaptivity_method = 0,
        itable = nothing,
        etable = nothing,
        prec = nothing,
        psetup = nothing,
        prec_side = 0
    )
    if linear_solver == :Band && (jac_upper == 0 || jac_lower == 0)
        error("Banded solver must set the jac_upper and jac_lower")
    end
    if !(
            linear_solver in (
                :None,
                :Diagonal,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Linear solver choice not accepted.")
    end
    if !(
            mass_linear_solver in (
                :None,
                :Diagonal,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Mass Matrix Linear solver choice not accepted.")
    end
    return ARKODE{
        method,
        linear_solver,
        mass_linear_solver,
        typeof(stiffness),
        typeof(itable),
        typeof(etable),
        typeof(prec),
        typeof(psetup),
    }(
        stiffness,
        jac_upper,
        jac_lower,
        mass_upper,
        mass_lower,
        krylov_dim,
        mass_krylov_dim,
        max_hnil_warns,
        max_error_test_failures,
        max_nonlinear_iters,
        max_convergence_failures,
        predictor_method,
        nonlinear_convergence_coefficient,
        dense_order,
        order,
        set_optimal_params,
        crdown,
        dgmax,
        rdiv,
        msbp,
        itable,
        etable,
        prec,
        psetup,
        prec_side
    )
end

SciMLBase.alg_order(alg::ARKODE) = 5

# DAE Algorithms
"""
    IDA(; linear_solver = :Dense, jac_upper = 0, jac_lower = 0,
        krylov_dim = 0, max_order = 5, max_error_test_failures = 7,
        max_nonlinear_iters = 3, nonlinear_convergence_coefficient = 0.33,
        nonlinear_convergence_coefficient_ic = 0.0033, max_num_steps_ic = 5,
        max_num_jacs_ic = 4, max_num_iters_ic = 10, max_num_backs_ic = 100,
        use_linesearch_ic = true, init_all = false,
        max_convergence_failures = 10, prec = nothing, psetup = nothing)

IDA: This is the IDA method from the Sundials.jl package.

# Keyword Arguments

- `linear_solver`: linear solver backend; `:Dense` is the default.
- `jac_upper`, `jac_lower`: upper and lower half-bandwidths for banded
  solvers. Both must be nonzero when `linear_solver = :Band` or `:LapackBand`.
- `krylov_dim`: maximum Krylov subspace dimension for iterative linear solvers.
- `max_order`, `max_error_test_failures`, `max_nonlinear_iters`, and
  `max_convergence_failures`: IDA order, error, and iteration limits.
- `nonlinear_convergence_coefficient`: nonlinear convergence coefficient.
- `nonlinear_convergence_coefficient_ic`: coefficient used by initial
  condition consistency calculations.
- `max_num_steps_ic`, `max_num_jacs_ic`, `max_num_iters_ic`, and
  `max_num_backs_ic`: initial-condition calculation limits.
- `use_linesearch_ic`: whether to use line search during initialization.
- `init_all`: whether the consistency calculation may modify all initial values.
- `prec`, `psetup`: left preconditioner and optional setup functions.

# Fields

The fields `jac_upper`, `jac_lower`, `krylov_dim`, `max_order`,
`max_error_test_failures`, `nonlinear_convergence_coefficient`,
`max_nonlinear_iters`, `max_convergence_failures`,
`nonlinear_convergence_coefficient_ic`, `max_num_steps_ic`,
`max_num_jacs_ic`, `max_num_iters_ic`, `max_num_backs_ic`,
`use_linesearch_ic`, `init_all`, `prec`, and `psetup` store the corresponding
constructor options. `linear_solver` is a type parameter.

# Returns

An `IDA` algorithm object for use with `SciMLBase.solve` on DAE problems.

# Throws

An error is thrown when an unsupported `linear_solver` is selected or when
`:Band` is selected without both band-widths.

# Examples

## Linear Solvers

Note that the constructors for the Sundials algorithms take a main argument:
linearsolver - This is the linear solver which is used in the Newton iterations. The
choices are:

* :Dense - A dense linear solver.
* :Band - A solver specialized for banded Jacobians. If used, you must set the position of the upper and lower non-zero diagonals via jac_upper and jac_lower.
* :LapackDense - A version of the dense linear solver that uses the Julia-provided OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than :Dense on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
* :LapackBand - A version of the banded linear solver that uses the Julia-provided OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than :Band on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
* :GMRES - A GMRES method. Recommended first choice Krylov method
* :BCG - A Biconjugate gradient method.
* :PCG - A preconditioned conjugate gradient method. Only for symmetric linear systems.
* :TFQMR - A TFQMR method.
* :KLU - A sparse factorization method. Requires that the user specifies a Jacobian. The Jacobian must be set as a sparse matrix in the ODEProblem type.

Note that the preconditioner for iterative linear solvers (if supplied) should be a left
preconditioner.

Example:

```julia
IDA() # Newton + Dense solver
IDA(linear_solver=:Band,jac_upper=3,jac_lower=3) # Banded solver with nonzero diagonals 3 up and 3 down
IDA(linear_solver=:BCG) # Biconjugate gradient method
```

### Preconditioners

Note that here `prec` is a (left) preconditioner function
`prec(z,r,p,t,y,fy,gamma,delta)` where:

- `z`: the computed output vector
- `r`: the right-hand side vector of the linear system
- `p`: the parameters
- `t`: the current independent variable
- `du`: the current value of `f(u,p,t)`
- `gamma`: the `gamma` of `W = M - gamma*J`
- `delta`: the iterative method tolerance

and `psetup` is the preconditioner setup function for pre-computing Jacobian
information. Where:

- `p`: the parameters
- `t`: the current independent variable
- `resid`: the current residual
- `u`: the current state
- `du`: the current derivative of the state
- `gamma`: the `gamma` of `W = M - gamma*J`

`psetup` is optional when `prec` is set.

### Additional Options

See [the Sundials manual](https://computing.llnl.gov/sites/default/files/ida_guide-5.7.0.pdf)
for details on the additional options. The option `init_all` controls the initial condition
consistency routine. If the initial conditions are inconsistent (i.e. they do not satisfy the
implicit equation), `init_all=false` means that the algebraic variables and derivatives will
be modified in order to satisfy the DAE. If `init_all=true`, all initial conditions will be
modified to satisfy the DAE.
"""
struct IDA{LinearSolver, P, PS} <: SundialsDAEAlgorithm{LinearSolver}
    jac_upper::Int
    jac_lower::Int
    krylov_dim::Int
    max_order::Int
    max_error_test_failures::Int
    nonlinear_convergence_coefficient::Float64
    max_nonlinear_iters::Int
    max_convergence_failures::Int
    nonlinear_convergence_coefficient_ic::Float64
    max_num_steps_ic::Int
    max_num_jacs_ic::Int
    max_num_iters_ic::Int
    max_num_backs_ic::Int
    use_linesearch_ic::Bool
    init_all::Bool
    prec::P
    psetup::PS
end
function IDA(;
        linear_solver = :Dense,
        jac_upper = 0,
        jac_lower = 0,
        krylov_dim = 0,
        max_order = 5,
        max_error_test_failures = 7,
        max_nonlinear_iters = 3,
        nonlinear_convergence_coefficient = 0.33,
        nonlinear_convergence_coefficient_ic = 0.0033,
        max_num_steps_ic = 5,
        max_num_jacs_ic = 4,
        max_num_iters_ic = 10,
        max_num_backs_ic = 100,
        use_linesearch_ic = true,
        init_all = false,
        max_convergence_failures = 10,
        prec = nothing,
        psetup = nothing
    )
    if linear_solver == :Band && (jac_upper == 0 || jac_lower == 0)
        error("Banded solver must set the jac_upper and jac_lower")
    end
    if !(
            linear_solver in (
                :None,
                :Diagonal,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Linear solver choice not accepted.")
    end
    return IDA{linear_solver, typeof(prec), typeof(psetup)}(
        jac_upper,
        jac_lower,
        krylov_dim,
        max_order,
        max_error_test_failures,
        nonlinear_convergence_coefficient,
        max_nonlinear_iters,
        max_convergence_failures,
        nonlinear_convergence_coefficient_ic,
        max_num_steps_ic,
        max_num_jacs_ic,
        max_num_iters_ic,
        max_num_backs_ic,
        use_linesearch_ic,
        init_all,
        prec,
        psetup
    )
end

"""
    KINSOL(; linear_solver = :Dense, jac_upper = 0, jac_lower = 0,
        userdata = nothing, prec_side = 0, krylov_dim = 0,
        globalization_strategy = :None, maxsetupcalls = 0)

KINSOL is Sundials' Newton-Krylov nonlinear solver for nonlinear and
steady-state problems.

# Keyword Arguments

- `linear_solver`: linear solver backend; `:Dense` is the default.
- `jac_upper`, `jac_lower`: nonzero band counts required by `linear_solver = :Band`.
- `userdata`: value passed through to the nonlinear residual and preconditioner callbacks.
- `prec_side`: preconditioning side accepted by the selected iterative solver.
- `krylov_dim`: maximum Krylov subspace dimension for iterative solvers.
- `globalization_strategy`: either `:None` or `:LineSearch`.
- `maxsetupcalls`: maximum nonlinear iterations between setup calls.

# Fields

The fields `jac_upper`, `jac_lower`, `userdata`, `prec_side`, `krylov_dim`,
`globalization_strategy`, and `maxsetupcalls` store the corresponding
constructor options. `linear_solver` is a type parameter.

# Returns

A `KINSOL` algorithm object for use with `SciMLBase.solve` on nonlinear and steady-state
problems.

# Throws

An error is thrown when an unsupported `linear_solver` or
`globalization_strategy` is selected.

## Linear Solver Choices

- `:Dense`: A dense linear solver
- `:Band`: A solver specialized for banded Jacobians. If used, you must set the
  position of the upper and lower non-zero diagonals via `jac_upper` and
  `jac_lower`.
- `:LapackDense`: A version of the dense linear solver that uses the Julia-provided
  OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than
  `:Dense` on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
- `:LapackBand`: A version of the banded linear solver that uses the Julia-provided
  OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than
  `:Band` on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
- `:GMRES`: A GMRES method. Recommended first choice Krylov method.
- `:BCG`: A biconjugate gradient method
- `:PCG`: A preconditioned conjugate gradient method. Only for symmetric
  linear systems.
- `:TFQMR`: A TFQMR method.
- `:KLU`: A sparse factorization method. Requires that the user specify a
  Jacobian. The Jacobian must be set as a sparse matrix in the `ODEProblem`
  type.

## Globalization Strategy

- `:None`: No globalization strategy
- `:LineSearch`: A line search globalization strategy

## Other Options

- `maxsetupcalls`: Maximum number of nonlinear iterations that can be performed between
  calls to the preconditioner or Jacobian setup function.

# Examples

```julia
KINSOL(linear_solver = :GMRES, globalization_strategy = :LineSearch)
```
"""
struct KINSOL{LinearSolver} <: SundialsNonlinearSolveAlgorithm{LinearSolver}
    jac_upper::Int
    jac_lower::Int
    userdata::Any
    prec_side::Int
    krylov_dim::Int
    globalization_strategy::Symbol
    maxsetupcalls::Int
end

function KINSOL(;
        linear_solver = :Dense,
        jac_upper = 0,
        jac_lower = 0,
        userdata = nothing,
        prec_side = 0,
        krylov_dim = 0,
        globalization_strategy = :None,
        maxsetupcalls = 0
    )
    if !(
            linear_solver in (
                :None,
                :Dense,
                :LapackDense,
                :Band,
                :LapackBand,
                :BCG,
                :GMRES,
                :FGMRES,
                :PCG,
                :TFQMR,
                :KLU,
            )
        )
        error("Linear solver choice not accepted.")
    end
    if !(globalization_strategy in (:LineSearch, :None))
        error("Globalization strategy not accepted.")
    end
    return KINSOL{linear_solver}(
        jac_upper, jac_lower, userdata, prec_side, krylov_dim,
        globalization_strategy, maxsetupcalls
    )
end

method_choice(alg::SundialsODEAlgorithm{Method}) where {Method} = Method
method_choice(alg::SundialsDAEAlgorithm) = :Newton
function linear_solver(
        alg::SundialsODEAlgorithm{
            Method,
            LinearSolver,
        }
    ) where {
        Method,
        LinearSolver,
    }
    return LinearSolver
end
linear_solver(alg::SundialsDAEAlgorithm{LinearSolver}) where {LinearSolver} = LinearSolver
function linear_solver(
        alg::SundialsNonlinearSolveAlgorithm{
            LinearSolver,
        }
    ) where {
        LinearSolver,
    }
    return LinearSolver
end
