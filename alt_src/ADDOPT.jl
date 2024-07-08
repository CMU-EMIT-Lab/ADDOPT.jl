using ForwardDiff
using SparseArrays
using LinearAlgebra
using CUDA
using Printf



include("dynamics/Dynamics.jl")
include("costs/Cost.jl")
include("constraints/Constraint.jl")

include("utilities/utilities.jl")

include("trajectory.jl")
include("process.jl")
include("problem.jl")

include("solver/al_ilqr.jl")