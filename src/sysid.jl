using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
const MOI = MathOptInterface
import HSL_jll

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
    A, E = θ[1], θ[2]
    Gₖ = A * exp(-E / R / Tₖ)
    Gₖ₊₁ = A * exp(-E / R / Tₖ₊₁)
    return (1 - (1 - yₖ) * exp(-Gₖ * dt))/2 + (1 - (1 - yₖ) * exp(-Gₖ₊₁ * dt))/2
end

# Optimizer Setup

struct SYSID_Problem <: MOI.AbstractNLPEvaluator
    dts::Vector{Vector{Float64}}
    Ts::Vector{Vector{Float64}}
    Ys::Vector{Vector{Float64}}
    ȳ::Vector{Float64}
    y₀::Vector{Float64}
    H₀::Float64
    H₁::Float64

    function SYSID_Problem(dts, Ts, H̄, H₀, H₁)
        @assert length(Ts) == length(H̄)
        @assert length(Ts) == length(dts)

        ΔH = H₁ - H₀
        ȳ = (H̄ .- H₀) ./ ΔH
        Ys = [zeros(length(T)) for T in Ts]

        new(dts, Ts, Ys, ȳ, zeros(length(ȳ)), H₀, H₁)
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

function MOI.eval_objective(prob::SYSID_Problem, z)
    θ = z
    ȳ = prob.ȳ
    simulate!(prob, θ)
    yf = [Y[end] for Y in prob.Ys]

    return 0.5 * sum((yf - ȳ) .^ 2)
end

function MOI.eval_objective_gradient(prob::SYSID_Problem, grad_f, z)
    θ = z
    dts, Ts = prob.dts, prob.Ts
    Nx = length(prob.y₀)
    Nz = length(z)
    ȳ = prob.ȳ

    ∂yf∂θ = zeros(Nx, Nz)
    ∂f∂θ = zeros(Nz)
    ∂yₖ∂θ = zeros(Nz)
    ∂yₖ₊₁∂θ = zeros(Nz)

    simulate!(prob, θ)
    yf = [Y[end] for Y in prob.Ys]

    for i in 1:Nx
        Nk = length(Ts[i])
        Y = prob.Ys[i]

        for k in 1:(Nk-1)
            Tₖ, Tₖ₊₁ = Ts[i][k], Ts[i][k+1]
            dt = dts[i][k]
            yₖ = Y[k]

            ∂f∂θ = ForwardDiff.gradient((θ) -> isothermal_step(yₖ, Tₖ, Tₖ₊₁, dt, θ), θ)
            ∂f∂yₖ = ForwardDiff.derivative((yₖ) -> isothermal_step(yₖ, Tₖ, Tₖ₊₁, dt, θ), yₖ)

            @. ∂yₖ₊₁∂θ = ∂f∂θ + ∂f∂yₖ * ∂yₖ∂θ
            ∂yₖ∂θ .= ∂yₖ₊₁∂θ
        end

        ∂yf∂θ[i, :] .= ∂yₖ∂θ
    end
    # @show ∂yf∂θ
    # @show yf - ȳ
    @show yf

    grad_f .= transpose(transpose(yf - ȳ) * ∂yf∂θ)
end

# function MOI.eval_constraint(prob::SYSID_Problem, c, z)
# end

# function MOI.eval_constraint_jacobian(prob::SYSID_Problem, jac, z)
# end

# MOI.jacobian_structure(prob::SYSID_Problem) = prob.constraint_jacobian_sparsity

# function MOI.hessian_lagrangian_structure(prob::SYSID_Problem)
#     return _
# end

# function MOI.eval_hessian_lagrangian(prob::SYSID_Problem, H, z, σ, μ)
# end

function MOI.features_available(prob::SYSID_Problem)
    return [:Grad]#, :Jac] #, :Hess]
end

MOI.initialize(prob::SYSID_Problem, features) = nothing

# Optimizing function

function fit_avrami(problem::SYSID_Problem, θ₀;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, solv="ma97")
    Nz = length(θ₀)
    gt = zeros(Nz)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["hsllib"] = HSL_jll.libhsl_path
    solver.options["linear_solver"] = solv

    @show Nz
    println("Checking objective function...")
    @time MOI.eval_objective(problem, θ₀)
    @time MOI.eval_objective(problem, θ₀)
    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, θ₀)
    @time MOI.eval_objective_gradient(problem, gt, θ₀)
    @show gt

    nlp_bounds = MOI.NLPBoundsPair.([], [])
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)
    z = MOI.add_variables(solver, Nz)

    for i in 1:lastindex(θ₀)
        MOI.set(solver, MOI.VariablePrimalStart(), z[i], θ₀[i])
        MOI.add_constraint(solver, z[i], MOI.GreaterThan(0.))
    end

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