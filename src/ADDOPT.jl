module ADDOPT

# Import dependencies
using LinearAlgebra
using SparseArrays
using StaticArrays

using Statistics
using StatsBase

using ForwardDiff

using CUDA
using ExponentialUtilities

using Printf
using ProgressMeter

# Export module names
export Dynamics
export nx, nxₖ₊₁, nu, Δt
export rate!, transition!

export Cost
export value

export Constraint
export nc, nc_eq, nc_ineq
export constraint!

export Trajectory, ConstraintTrajectory, Process
export rollout!

export Problem
export eval_cost, eval_lagrangian_cost, constraint_violation, eval_constraints!, eval_penalty_multiplier!
export al_ddp!

export powerfield_to_sequence, powerfield_to_sequence_with_traverse

export finite_diff!, path_to_points
export refine_grid
export get_coordinates
export sequence_to_realized_powerfield

# Include module files
include("dynamics/Dynamics.jl")
include("costs/Cost.jl")
include("constraints/Constraint.jl")

include("utilities/utilities.jl")

include("trajectory.jl")
include("process.jl")
include("problem.jl")

include("solver/al_ddp.jl")

include("instantiate.jl")

end