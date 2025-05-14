export WAAM_FiniteVolume
struct WAAM_FiniteVolume{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    nvox::Int

    A::M
    B::M
    e::V

    Ad::M
    Bd::M
    ed::V

    Δt::T

    function WAAM_FiniteVolume(A, B, e, Δt, c2c, T, V, M)
        nvox, _ = size(B)

        Ad, Bd, ed = discretize_linear_dynamics(A, B, e, Δt)

        new{T,V,M}(nvox, A, B, e, Ad * c2c, Bd, ed, Δt)
    end
end

nx(dynamics::WAAM_FiniteVolume)::Int = size(dynamics.Ad, 1)
nu(dynamics::WAAM_FiniteVolume)::Int = size(dynamics.Bd, 2)
Δt(dynamics::WAAM_FiniteVolume, u) = dynamics.Δt

function rate!(dynamics::WAAM_FiniteVolume{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    A, B = dynamics.A, dynamics.B

    mul!(ẋ, A, x, 1.0, 1.0)
    mul!(ẋ, B, u, 1.0, 1.0)
end

function transition!(dynamics::WAAM_FiniteVolume, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    Ad, Bd, ed = dynamics.Ad, dynamics.Bd, dynamics.ed

    xₖ₊₁ .= ed
    mul!(xₖ₊₁, Ad, xₖ, 1.0, 1.0)
    mul!(xₖ₊₁, Bd, uₖ, 1.0, 1.0)
end

function transition_state_jacobian!(dynamics::WAAM_FiniteVolume, A, xₖ, uₖ)
    A .= dynamics.Ad
end

function transition_input_jacobian!(dynamics::WAAM_FiniteVolume, B, xₖ, uₖ)
    B .= dynamics.Bd
end
