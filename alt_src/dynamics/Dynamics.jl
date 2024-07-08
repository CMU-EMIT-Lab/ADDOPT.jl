
abstract type Dynamics{T}
end

nx(dynamics::Dynamics)::Int = 0
nxₖ₊₁(dynamics::Dynamics)::Int = nx(dynamics)
nu(dynamics::Dynamics)::Int = 0
Δt(dynamics::Dynamics, u) = 0.0

function rate!(dynamics::Dynamics, ẋ, x, u)
    ẋ .= 0
end

function transition!(dynamics::Dynamics, xₖ₊₁, xₖ, uₖ)
    euler_step!(dynamics, xₖ₊₁, xₖ, uₖ, Δt(dynamics, uₖ))
    # rk4_step!(dynamics, xₖ₊₁, xₖ, uₖ, Δt(dynamics, uₖ))
end

function transition_state_jacobian!(dynamics::Dynamics, A, xₖ, uₖ)
    r = copy(xₖ)
    ForwardDiff.jacobian!(A, (xₖ₊₁, xₖ) -> transition!(dynamics, xₖ₊₁, xₖ, uₖ), r, xₖ)
end

function transition_input_jacobian!(dynamics::Dynamics, B, xₖ, uₖ)
    r = copy(xₖ)
    ForwardDiff.jacobian!(B, (xₖ₊₁, uₖ) -> transition!(dynamics, xₖ₊₁, xₖ, uₖ), r, uₖ)
end

include("furnace.jl")
include("tempering.jl")
include("furnace_tempering.jl")
include("pbf_powerfield.jl")