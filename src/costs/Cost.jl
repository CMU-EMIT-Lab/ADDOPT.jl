
abstract type Cost{T}
end

function value(cost::Cost{T}, x, u)::Float64 where {T<:AbstractFloat}
    return 0.0
end

function cost_state_gradient!(cost::Cost, ∂l∂x, x, u)
    ForwardDiff.gradient!(∂l∂x, (x) -> value(cost, x, u), x)
end

function cost_input_gradient!(cost::Cost, ∂l∂u, x, u)
    ForwardDiff.gradient!(∂l∂u, (u) -> value(cost, x, u), u)
end

function cost_state_hessian!(cost::Cost, ∂²l∂x², x, u)
    ForwardDiff.hessian!(∂²l∂x², (x) -> value(cost, x, u), x)
end

function cost_input_hessian!(cost::Cost, ∂²l∂u², x, u)
    ForwardDiff.hessian!(∂²l∂u², (u) -> value(cost, x, u), u)
end

function cost_input_state_hessian!(cost::Cost, ∂²l∂u∂x, x, u)
    ForwardDiff.jacobian!(∂²l∂u∂x, (g, x) -> ForwardDiff.gradient!(g, (u) -> value(cost, x, u), u), x)
end

include("quadratic.jl")