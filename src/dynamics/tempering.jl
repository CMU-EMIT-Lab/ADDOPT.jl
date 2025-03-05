
export Tempering
struct Tempering{T} <: Dynamics{T}
    lnA::T # Constant rate log
    E::T # Activation energy (kJ/mol)
    n::T # Avrami exponent
    R::T # Ideal gas constant # kJ⋅mol^−1⋅K^−1. 
    num::Int

    Δt::T

    function Tempering(lnA::T, n::T, E::T, Δt::T; num=1, R=8.3144598e-3) where {T}
        return new{T}(lnA, E, n, R, num, Δt)
    end
end

nx(dynamics::Tempering)::Int = dynamics.num
nu(dynamics::Tempering)::Int = dynamics.num
Δt(dynamics::Tempering, u) = dynamics.Δt


function rate!(dynamics::Tempering, ẋ::AbstractVector{E}, x, u) where {E}
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    T = u

    map!((x, T) -> n * exp(lnA - Q / R / T - x / n), ẋ, x, T)
end

function transition!(dynamics::Tempering, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)

    map!((yₖ, Tₖ) -> n * log(exp(yₖ / n) + exp(lnA - Q / R / Tₖ) * dt), xₖ₊₁, xₖ, uₖ)
end

function transition_state_jacobian!(dynamics::Tempering, A, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)
    A_diag = @view A[1:(size(A, 1)+1):end]

    map!((yₖ, Tₖ) -> 1.0 / (1.0 + exp(lnA - Q / R / Tₖ - yₖ / n) * dt), A_diag, xₖ, uₖ)
end

function transition_input_jacobian!(dynamics::Tempering, B, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)
    B_diag = @view B[1:(size(B, 1)+1):end]

    map!((yₖ, Tₖ) -> (n * Q / R / (Tₖ^2)) / (1.0 + exp(-lnA + Q / R / Tₖ + yₖ / n) / dt), B_diag, xₖ, uₖ)
end

function transition_state_jacobian_product_state_jacobian!(dynamics::Tempering, ∂Ap∂x, p, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)
    ∂Ap∂x_diag = @view ∂Ap∂x[1:(size(∂Ap∂x, 1)+1):end]

    ∂Ap∂x .= 0
    map!((y, T) -> (exp(Q / (R * T) + y / n - lnA) / dt) / (n * (exp(Q / (R * T) + y / n - lnA) / dt + 1)^2), ∂Ap∂x_diag, xₖ, uₖ)
    ∂Ap∂x_diag .*= p
end

function transition_input_jacobian_product_input_jacobian!(dynamics::Tempering, ∂Bp∂u, p, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)
    ∂Bp∂u_diag = @view ∂Bp∂u[1:(size(∂Bp∂u, 1)+1):end]

    ∂Bp∂u .= 0
    map!((y, T) -> n*dt*(Q / R / T)*(((Q * exp(-lnA + y/n + (Q / R / T)))/(R * (T^3) * (dt + exp(-lnA + y/n + (Q / R / T)))^2)) - ((2exp(lnA - (Q / R / T) - y/n))/((T^2)*(dt*exp(lnA - (Q / R / T) - y/n) + 1)))), ∂Bp∂u_diag, xₖ, uₖ)
    ∂Bp∂u_diag .*= p
end

function transition_input_jacobian_product_state_jacobian!(dynamics::Tempering, ∂Bp∂x, p, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)
    ∂Bp∂x_diag = @view ∂Bp∂x[1:(size(∂Bp∂x, 1)+1):end]

    ∂Bp∂x .= 0
    map!((y, T) -> -((exp(lnA - Q / (R * T) - y / n) * Q / dt) / (R * (1/dt + exp(lnA - Q / (R * T) - y / n))^2 * T^2)), ∂Bp∂x_diag, xₖ, uₖ)
    ∂Bp∂x_diag .*= p
end