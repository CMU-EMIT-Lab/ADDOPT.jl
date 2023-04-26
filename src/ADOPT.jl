module ADOPT

using MathOptInterface
using Ipopt
using ForwardDiff
const MOI = MathOptInterface

abstract type Dynamics end

struct InputDynamics <: Dynamics
    dynamics_function::Function
    input_function::Function
    parameters

    dynamics_jacobian::Union{Function,Matrix,Nothing}
    input_jacobian::Union{Function,Matrix,Nothing}

    dynamics_jacobian_sparsity::Union{Function,Matrix,Nothing}
    input_jacobian_sparsity::Union{Function,Matrix,Nothing}

    Nu::Int
    input_min::Vector
    input_max::Vector

    Nr::Int
    state_min::Vector
    state_max::Vector
end

struct TransferDynamics <: Dynamics
    dynamics_function::Function
    parameters
    jacobian::Union{Function,Matrix,Nothing}
    jacobian_sparsity::Union{Function,Matrix,Nothing}

    Ns::Int # 1 for just Temp, 2 for Temp + Mass
    state_min::Vector
    state_max::Vector
end

struct PropertyDynamics <: Dynamics
    dynamics_function::Function
    parameters
    jacobian::Union{Function,Matrix,Nothing}
    jacobian_sparsity::Union{Function,Matrix,Nothing}

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

    dynamics_constraint
    constraint_jacobian
    constraint_jacobian_sparsity

    idx

    function AdditiveProblem(process, objective, Nx, Ny, l, Nkb, Nc;
        (objective_gradient!)=nothing, Δt=nothing, dynamics_constraint=collocation)
        Nu, Nr, Ns, Nα = process.input_dynamics.Nu, process.input_dynamics.Nr, process.transfer_dynamics.Ns, process.property_dynamics.Nα
        free_time = prob.Δt === nothing
        idx = generate_z_indices(Nx, Ny, Nkb, Nc, Nu, Nr, Ns, Nα, free_time=free_time)

        constraint_jacobian =
        constraint_jacobian_sparsity =

        new(process, objective, objective_gradient!, Nx, Ny, l, Δt, Nkb, Nc,
            dynamics_constraint, constraint_jacobian, constraint_jacobian_sparsity, idx)
    end
end

function generate_z_indices(Nx, Ny, Nkb, Nc, Nu, Nr, Ns, Nα; free_time=false)
    NΔt = free_time ? 1 : 0
    Nstates = Nx * Ny * (Ns + Nα)
    Nknotvals = Nstates + Nr + Nu
    Npercycle = (NΔt + Nknotvals * Nkb + 1)
    Nz = Npercycle * Nc + Nstates # Last term is for state after final cooling

    s = [[(NΔt+1):(Nx*Ny*Ns+NΔt).+Nknotvals*(k-1)+Npercycle*(c-1) for k in 1:Nkb] for c in 1:Nc] # z[s[c][k]] gives s_(c,k), vector
    α = [[(Nx*Ny*Ns+NΔt+1):(Nstates+NΔt).+Nknotvals*(k-1)+Npercycle*(c-1) for k in 1:Nkb] for c in 1:Nc] # z[α[c][k]] gives α_(c,k), vector
    r = [[(Nstates+NΔt+1):(Nstates+NΔt+Nr).+Nknotvals*(k-1)+Npercycle*(c-1) for k in 1:Nkb] for c in 1:Nc] # z[r[c][k]] gives r_(c,k), vector
    u = [[(Nstates+NΔt+Nr+1):(Nknotvals).+Nknotvals*(k-1)+Npercycle*(c-1) for k in 1:Nkb] for c in 1:Nc] # z[u[c][k]] gives u_(c,k), vector
    Δt = free_time ? [1 + Npercycle * (c - 1) for c in 1:Nc] : nothing  # z[Δt[c]] gives Δt_c, scalar
    tc = [Npercycle * c for c in 1:Nc]                                # z[tc[c]] gives tc_c, scalar

    Nconstr = (Nstates + Nr)*Nkb*Nc

    return (Nz=Nz, Nstates=Nstates, s=s, α=α, r=r, u=u, Δt=Δt, tc=tc, Ncontsr)
end


function constraint_jacobian_sparsity(problem::AdditiveProblem)

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


    # Initial state


    # Final state?
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)

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