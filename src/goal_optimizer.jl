using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
using SparseArrays

struct ThermalICProblem{ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics} <: MOI.AbstractNLPEvaluator
    process::Process{ID,TD,PD}

    nvox::Int
    x₀::Vector{Float64}
    y₀::Vector{Float64}
    Eₘᵢₙ::Vector{Float64}
    Eₘₐₓ::Vector{Float64}

    Δt::Float64
    Nk::Int

    ȳ::Vector{Float64}
    Q::Diagonal{Float64,Vector{Float64}}
    J::Matrix{Float64}
end


function thermal_ic_to_property_final(problem::ThermalICProblem, E::Vector{Ty})::Vector{Ty} where Ty
    x, y0 = problem.x₀, problem.y₀
    Δt, Nk = problem.Δt, problem.Nk
    process = problem.process
    nvox = problem.nvox

    xi::Vector{Ty} = [E; x; y0]
    xf = inplace_solve_rk4(process, xi, Δt, Nk, 0.0, 0)

    return xf[(2nvox+1):3nvox]
end

function lsq_jac!(problem::ThermalICProblem, J::Matrix{Float64}, z::Vector{Float64})
    ForwardDiff.jacobian!(J, E -> thermal_ic_to_property_final(problem, E), z)
end


function MOI.eval_objective(prob::ThermalICProblem, z)
    y = thermal_ic_to_property_final(prob, z)
    e = @. y - prob.ȳ

    return 0.5 * dot(e, prob.Q, e)
end

function MOI.eval_objective_gradient(prob::ThermalICProblem, grad_f, z)
    y = thermal_ic_to_property_final(prob, z)
    e = @. y - prob.ȳ
    lsq_jac!(prob, prob.J, z)

    grad_f .= prob.J' * prob.Q * e
end

function MOI.eval_constraint(prob::ThermalICProblem, c, z)
end

function MOI.eval_constraint_jacobian(prob::ThermalICProblem, jac, z)
end

function MOI.hessian_lagrangian_structure(prob::ThermalICProblem)
    nvox = prob.nvox
    rows = (1:nvox) * ones(Int, nvox)'
    cols = ones(Int, nvox) * (1:nvox)'

    rows = reshape(rows, nvox^2)
    cols = reshape(cols, nvox^2)

    return collect(zip(rows, cols))
end

function MOI.eval_hessian_lagrangian(prob::ThermalICProblem, H, z, σ, μ)
    nvox = prob.nvox
    J = prob.J
    lsq_jac!(prob, J, z)
    Hmat = reshape(view(H, :), (nvox, nvox))
    Hmat .= J' * prob.Q * J

    H .*= σ
end

function MOI.features_available(prob::ThermalICProblem)
    return [:Grad, :Jac]#, :Hess]
end

MOI.initialize(prob::ThermalICProblem, features) = nothing
MOI.jacobian_structure(prob::ThermalICProblem) = []


function optimize_thermal_ic(problem::ThermalICProblem;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing, solv="ma97")

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["hsllib"] = HSL_jll.libhsl_path
    solver.options["linear_solver"] = solv

    nvox = problem.nvox
    if isnothing(z₀)
        z₀ = 1 .* problem.Eₘₐₓ
    end

    gt = zeros(nvox)
    μ0 = []

    println("Checking objective function...")
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)

    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)

    # println("Checking lagrangian hessian...")
    # structure = MOI.hessian_lagrangian_structure(problem)
    # H0 = zeros(length(structure))
    # @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
    # @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)

    nlp_bounds = MOI.NLPBoundsPair.([], [])
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, nvox)

    # Set primal bounds and initial values
    MOI.add_constraints(solver, z, MOI.LessThan.(problem.Eₘₐₓ))
    MOI.add_constraints(solver, z, MOI.GreaterThan.(problem.Eₘᵢₙ))

    for i in 1:nvox
        MOI.set(solver, MOI.VariablePrimalStart(), z[i], z₀[i])
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    λ = MOI.get(solver, MOI.NLPBlockDual())

    return result
end