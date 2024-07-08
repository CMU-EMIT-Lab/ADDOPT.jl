
struct Furnace{T} <: Dynamics{T}
    h::T
    T∞::T
    m::T
    cₚ::T
    A::T
    Δt::T
end

nx(dynamics::Furnace)::Int = 1
nu(dynamics::Furnace)::Int = 1
Δt(dynamics::Furnace, u) = dynamics.Δt

function rate!(dynamics::Furnace{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    P = u
    m, h, T∞, cₚ, A = dynamics.m, dynamics.h, dynamics.T∞, dynamics.cₚ, dynamics.A

    @. ẋ = (h * A / m / cₚ) * (T∞ - x) + P / (m * cₚ)
end

function transition!(dynamics::Furnace, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    m, h, T∞, cₚ, A = dynamics.m, dynamics.h, dynamics.T∞, dynamics.cₚ, dynamics.A
    dt = Δt(dynamics, uₖ)

    @. xₖ₊₁ = T∞ + uₖ / (h * A) + (xₖ - T∞ - uₖ / (h * A)) * exp(-h * A / m / cₚ * dt)
end

function transition_state_jacobian!(dynamics::Furnace, A, xₖ, uₖ)
    m, h, T∞, cₚ, Area = dynamics.m, dynamics.h, dynamics.T∞, dynamics.cₚ, dynamics.A
    dt = Δt(dynamics, uₖ)

    A .= exp(-h * Area / m / cₚ * dt)
end

function transition_input_jacobian!(dynamics::Furnace, B, xₖ, uₖ)
    m, h, T∞, cₚ, Area = dynamics.m, dynamics.h, dynamics.T∞, dynamics.cₚ, dynamics.A
    dt = Δt(dynamics, uₖ)

    B .= (1.0 - exp(-h * Area / m / cₚ * dt)) / (h * Area)
end
