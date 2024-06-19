using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
const MOI = MathOptInterface
import HSL_jll
using SparseArrays: findnz, nnz, nonzeros, sparse

const R = 8.3144598e-3 # kJ⋅mol^−1⋅K^−1. 

function dynamics(y, T, θ)
    A, E = θ[1], θ[2]
    return (1 - y) * A * exp(-E / R / T)
end

function rk4_step(yₖ, Tₖ, Tₖ₊₁, dt, θ)
    x₁ = yₖ
    k₁ = dynamics(x₁, Tₖ, θ)

    x₂ = yₖ + k₁ * dt / 2
    k₂ = dynamics(x₂, (Tₖ + Tₖ₊₁) / 2, θ)

    x₃ = yₖ + k₂ * dt / 2
    k₃ = dynamics(x₃, (Tₖ + Tₖ₊₁) / 2, θ)

    x₄ = yₖ + k₃ * dt
    k₄ = dynamics(x₄, Tₖ₊₁, θ)

    yₖ₊₁ = yₖ + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt

    return yₖ₊₁
end

function isothermal_step(yₖ, Tₖ, Tₖ₊₁, dt, θ)
    lnA, nd, E = θ
    n = 1 / nd
    lnA = lnA / n
    E = E / n
    Gₖ = exp(lnA - E / R / Tₖ)
    # @show lnA, n , E
    # Gₖ₊₁ = exp(lnA - E / R / Tₖ₊₁)

    #return ((yₖ^(1 / n) + Gₖ * dt)^n) / 2 + ((yₖ^(1 / n) + Gₖ₊₁ * dt)^n) / 2
    return n * log(exp(yₖ / n) + Gₖ * dt)
end

# Optimizer Setup

struct SYSID_Problem <: MOI.AbstractNLPEvaluator
    dts::Vector{Vector{Float64}}
    Ts::Vector{Vector{Float64}}
    Ys::Vector{Vector{Float64}}
    Ysd::Vector{Vector{ForwardDiff.Dual}}
    ȳ::Vector{Float64}
    y₀::Vector{Float64}

    function SYSID_Problem(dts, Ts, ȳ; y₀=nothing)
        @assert length(Ts) == length(ȳ)
        @assert length(Ts) == length(dts)

        # ΔH = H₁ - H₀
        # ȳ = (H̄ .- H₀) ./ ΔH
        Ys = [zeros(length(T)) for T in Ts]
        Ysd = [zeros(ForwardDiff.Dual, length(T)) for T in Ts]
        if isnothing(y₀)
            y₀ = zeros(length(ȳ))
        end
        for i in 1:length(ȳ)
            Ys[i][1] = y₀[i]
            Ysd[i][1] = y₀[i]
        end

        new(dts, Ts, Ys, Ysd, ȳ, y₀)
    end
end

function simulate!(prob::SYSID_Problem, θ)
    dts, Ts = prob.dts, prob.Ts
    Nx = length(prob.y₀)

    for i in 1:Nx
        Nk = length(Ts[i])
        Y = prob.Ys[i]

        for k in 1:(Nk-1)
            Y[k+1] = isothermal_step(Y[k], Ts[i][k], Ts[i][k+1], dts[i][k], θ)
        end
    end
end

function _simulate(prob::SYSID_Problem, θ)
    y = zeros(eltype(θ), length(prob.y₀))
    y .= log.(-log.(1 .- prob.y₀))
    dts, Ts = prob.dts, prob.Ts
    Nx = length(prob.y₀)
    flush(stdout)

    for i in 1:Nx
        Nk = length(Ts[i])

        for k in 1:(Nk-1)
            y[i] = isothermal_step(y[i], Ts[i][k], Ts[i][k+1], dts[i][k], θ)
        end
    end

    return y
end

function simulate(prob::SYSID_Problem, θ)
    y = _simulate(prob, θ)
    return 1 .- exp.(-exp.(y))
end

function MOI.eval_objective(prob::SYSID_Problem, z)
    θ = z
    y₀, ȳ = prob.y₀, prob.ȳ
    zf = _simulate(prob, θ)

    z̄ = log.(-log.(1 .- ȳ)) #log.((-log.(1 .- ȳ)) .^ (1 / n) .- (-log.(1 .- y₀)) .^ (1 / n)) # 
    # zf = log.(-log.(1 .- yf)) #log.((-log.(1 .- yf)) .^ (1 / n) .- (-log.(1 .- y₀)) .^ (1 / n))# 

    flush(stdout)
    if isdefined(Main, :IJulia)
        Main.IJulia.stdio_bytes[] = 0
    end

    return 0.5 * sum(((zf .- z̄)) .^ 2)
end

function MOI.eval_objective_gradient(prob::SYSID_Problem, grad_f, z)
    ForwardDiff.gradient!(grad_f, (z) -> MOI.eval_objective(prob::SYSID_Problem, z), z)
end

# function MOI.eval_constraint(prob::SYSID_Problem, c, z)
# end

# function MOI.eval_constraint_jacobian(prob::SYSID_Problem, jac, z)
# end

# MOI.jacobian_structure(prob::SYSID_Problem) = prob.constraint_jacobian_sparsity

function MOI.hessian_lagrangian_structure(prob::SYSID_Problem)
    r, c, _ = findnz(sparse(ones(3, 3)))
    return collect(zip(r, c))
end

function MOI.eval_hessian_lagrangian(prob::SYSID_Problem, H, z, σ, μ)
    h = ForwardDiff.hessian((z) -> MOI.eval_objective(prob::SYSID_Problem, z), z)
    H .= vec(h)
    H .*= σ
end

function MOI.features_available(prob::SYSID_Problem)
    return [:Grad, :Hess]
end

MOI.initialize(prob::SYSID_Problem, features) = nothing

# Optimizing function

function fit_avrami(problem::SYSID_Problem, θ₀, zₘᵢₙ, zₘₐₓ;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, solv="ma97")
    Nz = length(θ₀)
    gt = zeros(Nz)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["hsllib"] = HSL_jll.libhsl_path
    solver.options["linear_solver"] = solv
    solver.options["print_frequency_iter"] = 10
    solver.options["neg_curv_test_tol"] = 1e-11

    @show Nz
    println("Checking objective function...")
    @time MOI.eval_objective(problem, θ₀)
    @time MOI.eval_objective(problem, θ₀)
    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, θ₀)
    @time MOI.eval_objective_gradient(problem, gt, θ₀)
    @show gt
    h = ForwardDiff.hessian((z) -> MOI.eval_objective(problem, z), θ₀)
    display(h)
    display(eigen(h))

    nlp_bounds = MOI.NLPBoundsPair.([], [])
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)
    z = MOI.add_variables(solver, Nz)

    for i in 1:lastindex(θ₀)
        MOI.set(solver, MOI.VariablePrimalStart(), z[i], θ₀[i])
    end
    MOI.add_constraints(solver, z, MOI.GreaterThan.(zₘᵢₙ))
    MOI.add_constraints(solver, z, MOI.LessThan.(zₘₐₓ))


    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    flush(stdout)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    return result
end

# dt = 10.0
# t = 1.0:dt:50
# Nk = length(t)
# Yr = zeros(Nk)
# Yi = zeros(Nk)
# T = 1600 * log.(t)
# for k in 1:(Nk-1)
#     Yr[k+1] =        rk4_step(Yr[k], T[k], T[k+1], dt, θ)
#     Yi[k+1] = isothermal_step(Yi[k], T[k], T[k+1], dt, θ)
# end
# plot()
# plot!(t, Yr, label="RK4")
# plot!(t, Yi, label="Isothermal")