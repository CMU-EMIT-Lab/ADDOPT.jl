module ADOPT

using MathOptInterface
using Ipopt

abstract type Dynamics end

struct InputDynamics <: Dynamics
    dynamics_function::Function
    input_function::Function
    
    dynamics_jacobian::Union{Function,Matrix,Nothing}
    input_jacobian::Union{Function,Matrix,Nothing}

    dynamics_jacobian_sparsity::Union{Function,Matrix,Nothing}
    input_jacobian_sparsity::Union{Function,Matrix,Nothing}

    Nu::Int
    Nr::Int
end

struct TransferDynamics <: Dynamics
    dynamics_function::Function
    jacobian::Union{Function,Matrix,Nothing}
    jacobian_sparsity::Union{Function,Matrix,Nothing}

    Nx::Int
end

struct PropertyDynamics <: Dynamics
    dynamics_function::Function
    jacobian::Union{Function,Matrix,Nothing}
    jacobian_sparsity::Union{Function,Matrix,Nothing}

    Nα::Int
end

struct Process
    input_dynamics::InputDynamics
    transfer_dynamics::TransferDynamics
    property_dynamics::PropertyDynamics
end

struct AdditiveProblem
    process::Process
    parameters

    objective::Function
    gradient::Union{Function,Nothing}
    hessian::Union{Function,Matrix,Nothing}

    Nx::Int
    Nz::Int
    l::Real

    Δt::Union{Real,Nothing}
    Nk_process
    N_cycles::Int
end

function optimize_trajectory(problem::AdditiveProblem, x₀, x̄, X₀, U₀)

end

end