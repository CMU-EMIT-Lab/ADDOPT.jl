
export QuadraticCost
struct QuadraticCost{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Cost{T}
    Q::M
    R::M
    x̄::V
    ū::V

    ex::V
    eu::V

    Qex::V
    Reu::V

    function QuadraticCost(Q::M, R::M, x̄::V, ū::V) where {T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}}
        ex = copy(x̄)
        eu = copy(ū)
        Qex = copy(ex)
        Reu = copy(eu)

        new{T,V,M}(Q, R, x̄, ū, ex, eu, Qex, Reu)
    end
end

function value(cost::QuadraticCost{T,V,M}, x, u)::Float64 where {T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    Q, R = cost.Q, cost.R
    x̄, ū = cost.x̄, cost.ū
    ex, eu = cost.ex, cost.eu
    Qex, Reu = cost.Qex, cost.Reu

    @. ex = x - x̄
    @. eu = u - ū

    mul!(Qex, Q, ex)
    mul!(Reu, R, eu)

    return 0.5 * (dot(ex, Qex) + dot(eu, Reu))
end

function cost_state_gradient!(cost::QuadraticCost, ∂l∂x, x, u)
    Q, R = cost.Q, cost.R
    x̄, ū = cost.x̄, cost.ū
    ex, eu = cost.ex, cost.eu

    @. ex = x - x̄
    mul!(∂l∂x, Q, ex)
end

function cost_input_gradient!(cost::QuadraticCost, ∂l∂u, x, u)
    Q, R = cost.Q, cost.R
    x̄, ū = cost.x̄, cost.ū
    ex, eu = cost.ex, cost.eu

    @. eu = u - ū
    mul!(∂l∂u, R, eu)
end

function cost_state_hessian!(cost::QuadraticCost, ∂²l∂x², x, u)
    Q, R = cost.Q, cost.R

    ∂²l∂x² .= Q
end

function cost_input_hessian!(cost::QuadraticCost, ∂²l∂u², x, u)
    Q, R = cost.Q, cost.R

    ∂²l∂u² .= R
end

function cost_input_state_hessian!(cost::QuadraticCost, ∂²l∂u∂x, x, u)
    ∂²l∂u∂x .= 0.0
end