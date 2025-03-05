struct WAAM_Voxelized{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    nvox::Int

    l::T

    k::T
    ρ::T
    cₚ::T

    T∞::T
    T₀::T

    h::T

    A::M
    B::M
    e::V

    Ad::M
    Bd::M
    ed::V

    Δt::T

    σ::T

    function WAAM_Voxelized(A, B, l, k, ρ, cₚ, T∞, T₀, h, Δt, T, V, M; σ=0.0, buffer=0)
        nvox, _ = size(B)
        α = k / ρ / cₚ
        C = ρ * l^3 * cₚ

        Ad, Bd, ed = discretize_linear_dynamics(A, B, e, Δt)

        new{T,V,M}(nvox, l, k, ρ, cₚ, T∞, T₀, h, A, B, e, Ad, Bd, ed, Δt, σ, buffer)
    end
end

nx(dynamics::WAAM_Voxelized)::Int = dynamics.nx * dynamics.ny * dynamics.nz
nu(dynamics::WAAM_Voxelized)::Int = (dynamics.nx - 2dynamics.buffer) * (dynamics.ny - 2dynamics.buffer)
Δt(dynamics::WAAM_Voxelized, u) = dynamics.Δt

function rate!(dynamics::WAAM_Voxelized{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    A, B = dynamics.A, dynamics.B

    mul!(ẋ, A, x, 1.0, 1.0)
    mul!(ẋ, B, u, 1.0, 1.0)
end

function transition!(dynamics::WAAM_Voxelized, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    Ad, Bd, ed = dynamics.Ad, dynamics.Bd, dynamics.ed

    xₖ₊₁ .= ed
    mul!(xₖ₊₁, Ad, xₖ, 1.0, 1.0)
    mul!(xₖ₊₁, Bd, uₖ, 1.0, 1.0)
end

function transition_state_jacobian!(dynamics::WAAM_Voxelized, A, xₖ, uₖ)
    A .= dynamics.Ad
end

function transition_input_jacobian!(dynamics::WAAM_Voxelized, B, xₖ, uₖ)
    B .= dynamics.Bd
end
