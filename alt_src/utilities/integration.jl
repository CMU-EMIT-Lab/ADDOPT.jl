
function rk4_step!(dynamics::D, xₖ₊₁::Vector{T}, xₖ, uₖ, Δt) where {T, D <: Dynamics}
    Nx = nx(dynamics)
    k₁, k₂, k₃, k₄ = zeros(T, Nx), zeros(T, Nx), zeros(T, Nx), zeros(T, Nx)
    x₁, x₂, x₃, x₄ = zeros(T, Nx), zeros(T, Nx), zeros(T, Nx), zeros(T, Nx)

    rk4_step!(dynamics, xₖ₊₁, xₖ, uₖ, Δt, x₁, x₂, x₃, x₄, k₁, k₂, k₃, k₄)
end

function rk4_step!(dynamics::D, xₖ₊₁, xₖ, uₖ, Δt, x₁, x₂, x₃, x₄, k₁, k₂, k₃, k₄) where D <: Dynamics
    x₁ .= xₖ
    rate!(dynamics, k₁, x₁, uₖ)

    @. x₂ = xₖ + k₁ * Δt / 2
    rate!(dynamics, k₂, x₂, uₖ)

    @. x₃ = xₖ + k₂ * Δt / 2
    rate!(dynamics, k₃, x₃, uₖ)

    @. x₄ = xₖ + k₃ * Δt
    rate!(dynamics, k₄, x₄, uₖ)

    @. xₖ₊₁ = xₖ + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * Δt
end

function euler_step!(dynamics::D, xₖ₊₁, xₖ, uₖ, Δt) where D <: Dynamics
    rate!(dynamics, xₖ₊₁, xₖ, uₖ)
    xₖ₊₁ .*= Δt
    xₖ₊₁ .+= xₖ
end