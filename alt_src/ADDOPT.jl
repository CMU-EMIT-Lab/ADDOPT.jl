using ForwardDiff
using SparseArrays
using LinearAlgebra
using CUDA
using Printf


import LinearAlgebra.mul!

function mul!(Y, A, B::Diagonal{T,CuVector{T}}) where {T}
    Y .= A
    Y .*= B.diag'
end

include("dynamics/Dynamics.jl")
include("costs/Cost.jl")
include("constraints/Constraint.jl")

include("utilities/utilities.jl")

include("trajectory.jl")
include("process.jl")
include("problem.jl")

include("solver/al_ilqr.jl")

include("instantiate.jl")