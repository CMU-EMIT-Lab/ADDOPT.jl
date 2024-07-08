
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

    map!((yₖ, Tₖ) -> 1.0 / (1.0 + exp(lnA - Q / R / Tₖ - yₖ / n) * dt), A, xₖ, uₖ)
end

function transition_input_jacobian!(dynamics::Tempering, B, xₖ, uₖ)
    lnA, Q, n, R = dynamics.lnA, dynamics.E, dynamics.n, dynamics.R
    dt = Δt(dynamics, uₖ)

    map!((yₖ, Tₖ) -> (n * Q / R / (Tₖ^2)) / (1.0 + exp(-lnA + Q / R / Tₖ + yₖ / n) / dt), B, xₖ, uₖ)
end