module ADOPT

using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
using Symbolics, SparseArrays, SparseDiffTools
using SparseArrays: findnz
const MOI = MathOptInterface

abstract type Dynamics end

struct InputDynamics <: Dynamics
    dynamics_function!::Function
    input_function!::Function
    parameters

    Nu::Int
    input_min::Vector
    input_max::Vector

    Nr::Int
    state_min::Vector
    state_max::Vector
end

struct TransferDynamics <: Dynamics
    dynamics_function!::Function
    parameters

    Ns::Int # 1 for just Temp, 2 for Temp + Mass
    state_min::Vector
    state_max::Vector
end

struct PropertyDynamics <: Dynamics
    dynamics_function!::Function
    parameters

    Nα::Int
    property_min::Vector
    property_max::Vector
end

struct Process
    input_dynamics::InputDynamics
    transfer_dynamics::TransferDynamics
    property_dynamics::PropertyDynamics
end

struct AdditiveProblem <: MOI.AbstractNLPEvaluator
    process::Process

    objective::Function
    objective_gradient!::Union{Function,Nothing}
    # hessian::Union{Function,Matrix,Nothing}

    Nx::Int
    Ny::Int
    l::Real

    Δt::Union{Real,Nothing}
    Nkb::Int # Number of knots per build cycle
    # Nkc::Int # Number of knots per cooling cycle
    Nc::Int  # Number of cycles

    dynamics_constraint!::Function
    constraint_jacobian_sparsity

    idx
    sparsity_cache
    colloc!
    jump_constraint!

    function AdditiveProblem(process, objective, Nx, Ny, l, Nkb, Nc;
        (objective_gradient!)=nothing, Δt=nothing)
        Nu, Nr, Ns, Nα = process.input_dynamics.Nu, process.input_dynamics.Nr, process.transfer_dynamics.Ns, process.property_dynamics.Nα
        free_time = prob.Δt === nothing
        idx = generate_z_indices(Nx, Ny, Nkb, Nc, Nu, Nr, Ns, Nα, free_time=free_time)

        dynamics_constraint! = (c, z) -> dynamics_constraint!(c, z, idx, process) # in-place calculation of constraints
        f!(ẋ, x, u) = combined_dynamics!(ẋ, x, u, process)
        

        colloc!(r, xₖ, xₖ₊₁, uₖ, uₖ₊₁, Δt) = collocation_constraint!(r, xₖ, xₖ₊₁, uₖ, uₖ₊₁, f!, Δt)
        jump_constraint!(r, xₖ, xₖ₊₁, Δt) = 

        constraint_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, colloc!, jump_constraint!::Function)

        new(process, objective, objective_gradient!, Nx, Ny, l, Δt, Nkb, Nc,
            dynamics_constraint!, constraint_jacobian_sparsity, idx, sparsity_cache,
            colloc!, jump_constraint!)
    end
end

function generate_z_indices(Nx, Ny, Nkb, Nc, Nu, Nr, Ns, Nα; free_time=false)
    NΔt = free_time ? 1 : 0
    Nstates = Nx * Ny * (Ns + Nα)
    Nknotvals = Nstates + Nr + Nu
    Npercycle = (NΔt + Nknotvals * Nkb + 1)
    Nz = Npercycle * Nc + Nstates # Last term is for state after final cooling

    s = [[((NΔt+1):(Nx*Ny*Ns+NΔt)) .+ Nknotvals * (k - 1) + Npercycle * (c - 1) for k in 1:Nkb] for c in 1:Nc] # z[s[c][k]] gives s_(c,k), vector
    α = [[((Nx*Ny*Ns+NΔt+1):(Nstates+NΔt)) .+ Nknotvals * (k - 1) + Npercycle * (c - 1) for k in 1:Nkb] for c in 1:Nc] # z[α[c][k]] gives α_(c,k), vector
    r = [[((Nstates+NΔt+1):(Nstates+NΔt+Nr)) .+ Nknotvals * (k - 1) + Npercycle * (c - 1) for k in 1:Nkb] for c in 1:Nc] # z[r[c][k]] gives r_(c,k), vector
    u = [[((Nstates+NΔt+Nr+1):(Nknotvals)) .+ Nknotvals * (k - 1) + Npercycle * (c - 1) for k in 1:Nkb] for c in 1:Nc] # z[u[c][k]] gives u_(c,k), vector
    x = [[((NΔt+1):(Nknotvals)) .+ Nknotvals * (k - 1) + Npercycle * (c - 1) for k in 1:Nkb] for c in 1:Nc]
    Δt = free_time ? [1 + Npercycle * (c - 1) for c in 1:Nc] : nothing  # z[Δt[c]] gives Δt_c, scalar
    tc = [Npercycle * c for c in 1:Nc]                                  # z[tc[c]] gives tc_c, scalar

    Nconstr = (Nstates + Nr) * Nkb * Nc

    return (Nz=Nz, Nstates=(Nstates + Nr), s=s, α=α, r=r, u=u, x=x, Δt=Δt, tc=tc, Nconstr=Ncontsr, Nkb=Nkb, Nc=Nc, Nu=Nu)
end

function combined_dynamics!(f::Vector, x::Vector, u::Vector, process::Process)
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics
    Ns, Nα = td.Ns, pd.Nα
    Nr, Nu = id.Nr, id.Nu

    s = @view x[1:Ns]
    α = @view x[(Ns+1):(Ns+Nα)]
    r = @view x[(Ns+Nα+1):end]

    ds = @view f[1:Ns]
    dα = @view f[(Ns+1):(Ns+Nα)]
    dr = @view f[(Ns+Nα+1):end]

    td.dynamics_function!(ds, s)
    id.input_function!(ds, r, u)
    pd.dynamics_function!(dα, α, s)
    id.dynamics_function!(dr, s, r, u)
end

function collocation_constraint!(r, xₖ, xₖ₊₁, uₖ, uₖ₊₁, f!, Δt)
    Nx = length(xₖ)

    fₖ = zeros(eltype(xₖ), Nx)
    fₖ₊₁ = zeros(eltype(xₖ), Nx)
    fₘ = zeros(eltype(xₖ), Nx)

    f!(fₖ, xₖ, uₖ)
    f(fₖ₊₁, xₖ₊₁, uₖ₊₁)

    xₘ = 0.5 * (xₖ + xₖ₊₁) + (Δt / 8.0) * (fₖ - fₖ₊₁)
    uₘ = 0.5 * (uₖ + uₖ₊₁)
    ẋₘ = (3 / (2 * Δt)) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    f(fₘ, xₘ, uₘ)

    @. r[:] = fₘ - ẋₘ
end

function dynamics_constraint!(c, z, idx, process::Process, Δt)
    Nc, Nkb = idx.Nc, idx.Nkb

    for c in 1:Nc
        for k in 1:(Nkb-1)
            xₖ = view(z, idx.s[c][k])
            xₖ₊₁ = view(z, idx.s[c][k+1])
            uₖ
            uₖ₊₁

        end
    end
end

function constraint_jacobian!(jac, z, idx, colloc!::Function, jump_constraint!::Function, sparsity_cache)
    Nc, Nk = idx.Nc, idx.Nkb
    colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color = sparsity_cache

    colloc_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt), xₖ, colorvec=colloc_∂uₖ_color)
    colloc_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt), uₖ, colorvec=colloc_∂xₖ_color)
    colloc_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, X, uₖ₊₁, Δt), xₖ₊₁, colorvec=colloc_∂xₖ₊₁_color)
    colloc_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, xₖ₊₁, X, Δt), uₖ₊₁, colorvec=colloc_∂uₖ₊₁_color)
    colloc_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X), Δt, colorvec=colloc_∂Δt_color)

    i = 0
    J∂xₖ = Float64.(sparse(colloc_∂xₖ_sparsity))
    J∂uₖ = Float64.(sparse(colloc_∂uₖ_sparsity))
    J∂xₖ₊₁ = Float64.(sparse(colloc_∂xₖ₊₁_sparsity))
    J∂uₖ₊₁ = Float64.(sparse(colloc_∂uₖ₊₁_sparsity))
    J∂Δt = Float64.(sparse(colloc_∂Δt_sparsity))
    for c in 1:Nc
        Δt = @view z[idx.Δt[c]]

        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k+1]]
            xₖ₊₁ = @view z[idx.x[c][k]]
            uₖ₊₁ = @view z[idx.u[c][k+1]]

            colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂Δt)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂xₖ)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂uₖ)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂xₖ₊₁)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            colloc_∂uₖ₊₁(J∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂uₖ₊₁)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)
        end

        # jump jacobian
    end
end

function constraint_jacobian_sparsity(idx, colloc!::Function, jump_constraint!::Function)
    Nx, Nu = idx.Nstates, idx.Nu
    Nk, Nc = idx.Nkb, idx.Nc

    rd = Vector{Float64}(undef, Nx)
    xd = Vector{Float64}(undef, Nx)
    ud = Vector{Float64}(undef, Nu)
    Δtd = Vector{Float64}(undef, 1)

    colloc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, X, ud, xd, ud, Δtd), rd, xd)
    colloc_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd, X, xd, ud, Δtd), rd, ud)
    colloc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd, ud, X, ud, Δtd), rd, xd)
    colloc_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd, ud, xd, X, Δtd), rd, ud)
    colloc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, Δt) -> colloc!(r, xd, ud, xd, ud, X), rd, Δtd)

    colloc_∂uₖ_color = matrix_colors(Float64.(colloc_∂uₖ_sparsity))
    colloc_∂xₖ_color = matrix_colors(Float64.(colloc_∂xₖ_sparsity))
    colloc_∂xₖ₊₁_color = matrix_colors(Float64.(colloc_∂xₖ₊₁_sparsity))
    colloc_∂uₖ₊₁_color = matrix_colors(Float64.(colloc_∂uₖ₊₁_sparsity))
    colloc_∂Δt_color = matrix_colors(Float64.(colloc_∂Δt_sparsity))

    row_offset = 0
    col_offset = 0
    rows = []
    cols = []
    for c in 1:Nc
        for k in 1:(Nk-1)
            r, c, _ = findnz(colloc_∂Δt_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c)
            col_offset = 1 + (Nx + Nu) * (k - 1) + ((Nx + Nu) * (Nk - 1)) * (c - 1)

            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(colloc_∂uₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nu

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(colloc_∂uₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nu

            row_offset += Nx
        end

        ## jump sparsity
    end

    total_structure = collect(zip(rows, cols))
    sparsity_cache = colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color
    return total_structure, sparsity_cache
end


function MOI.eval_objective(prob::AdditiveProblem, z)
    return prob.objective(z)
end

function MOI.eval_objective_gradient(prob::AdditiveProblem, grad_f, z)
    if !isnothing(prob.gradient)
        prob.objective_gradient!(grad_f, z)
    else
        ForwardDiff.gradient!(grad_f, prob.objective, z)
    end
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    # Collocation
    prob.dynamics_constraint!(c, z)

    # Initial state


    # Final state?
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    constraint_jacobian!(jac, z, prob.idx, prob.colloc!, prob.jump_constraint!, prob.sparsity_cache)
end

MOI.features_available(prob::AdditiveProblem) = [:Grad, :Jac]
MOI.initialize(prob::AdditiveProblem, features) = nothing
MOI.jacobian_structure(prob::AdditiveProblem) = prob.constraint_jacobian_sparsity

function optimize_trajectory(problem::AdditiveProblem, x₀, x̄;
    X₀=nothing, U₀=nothing,
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    idx = problem.idx
    Nz, Nconstr = idx.Nz, idx.Nconstr
    process = problem.process
    Nx, Ny, Nkb, Nc = problem.Nx, problem.Ny, problem.Nkb, problem.Nc
    Nu, Nr, Ns, Nα = process.input_dynamics.Nu, process.input_dynamics.Nr, process.transfer_dynamics.Ns, process.property_dynamics.Nα
    if isnothing(X₀)
        X₀ = range(x₀, x̄, length=100)
    end

    if isnothing(U₀)
        U₀ = randn()
    end

    c_l = zeros(Nconstr)
    c_u = zeros(Nconstr)
    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, Nz)

    # Set primal bounds and initial values
    for c in 1:Nc
        for k in 1:Nkb
            for j in 1:Nr
                MOI.add_constraint(solver, z[idx.r[c][k]][j], MOI.LessThan(process.input_dynamics.state_min[j]))
                MOI.add_constraint(solver, z[idx.r[c][k]][j], MOI.GreaterThan(process.input_dynamics.state_max[j]))
                MOI.set(solver, MOI.VariablePrimalStart(), z[idx.r[c][k]][j], ...)
            end
            for j in 1:Nu
                MOI.add_constraint(solver, z[idx.u[c][k]][j], MOI.LessThan(process.input_dynamics.input_min[j]))
                MOI.add_constraint(solver, z[idx.u[c][k]][j], MOI.GreaterThan(process.input_dynamics.input_max[j]))
                MOI.set(solver, MOI.VariablePrimalStart(), z[idx.u[c][k]][j], ...)
            end
            for i in 1:Ny*Nx
                for j in 1:Ns
                    MOI.add_constraint(solver, z[idx.s[c][k]][(i-1)*Ns+j], MOI.LessThan(process.transfer_dynamics.state_min[j]))
                    MOI.add_constraint(solver, z[idx.s[c][k]][(i-1)*Ns+j], MOI.GreaterThan(process.transfer_dynamics.state_min[j]))
                    MOI.set(solver, MOI.VariablePrimalStart(), z[idx.s[c][k]][(i-1)*Ns+j], ...)
                end
                for j in 1:Nα
                    MOI.add_constraint(solver, z[idx.α[c][k]][(i-1)*Nα+j], MOI.LessThan(process.property_dynamics.property_min[j]))
                    MOI.add_constraint(solver, z[idx.α[c][k]][(i-1)*Nα+j], MOI.GreaterThan(process.property_dynamics.property_max[j]))
                    MOI.set(solver, MOI.VariablePrimalStart(), z[idx.α[c][k]][(i-1)*Nα+j], ...)
                end
            end
        end
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
end

end