export PBFPowerField
struct PBFPowerField{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    nx::Int
    ny::Int
    nz::Int

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
    buffer::Int

    function PBFPowerField(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, h, Δt, T, V, M; σ=0.0, buffer=0)
        α = k / ρ / cₚ
        C = ρ * l^3 * cₚ

        A, B, e = matrices_for_voxel_conduction(nx, ny, nz, l, α, C, h, T₀, T∞, σ, buffer)
        Ad, Bd, ed = discretize_linear_dynamics(A, B, e, Δt)

        new{T,V,M}(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, h, A, B, e, Ad, Bd, ed, Δt, σ, buffer)
    end
end

nx(dynamics::PBFPowerField)::Int = dynamics.nx * dynamics.ny * dynamics.nz
nu(dynamics::PBFPowerField)::Int = (dynamics.nx - 2dynamics.buffer) * (dynamics.ny - 2dynamics.buffer)
Δt(dynamics::PBFPowerField, u) = dynamics.Δt

function rate!(dynamics::PBFPowerField{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    A, B = dynamics.A, dynamics.B

    mul!(ẋ, A, x, 1.0, 1.0)
    mul!(ẋ, B, u, 1.0, 1.0)
end

function transition!(dynamics::PBFPowerField, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    Ad, Bd, ed = dynamics.Ad, dynamics.Bd, dynamics.ed

    xₖ₊₁ .= ed
    mul!(xₖ₊₁, Ad, xₖ, 1.0, 1.0)
    mul!(xₖ₊₁, Bd, uₖ, 1.0, 1.0)
end

function transition_state_jacobian!(dynamics::PBFPowerField, A, xₖ, uₖ)
    A .= dynamics.Ad
end

function transition_input_jacobian!(dynamics::PBFPowerField, B, xₖ, uₖ)
    B .= dynamics.Bd
end