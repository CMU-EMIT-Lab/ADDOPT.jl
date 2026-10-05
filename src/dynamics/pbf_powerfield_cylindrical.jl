export PBFPowerFieldCylindrical
struct PBFPowerFieldCylindrical{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    nz::Int
    nr::Int

    Δr::T
    Δz::T

    k::V
    ρ::V
    cₚ::V

    T∞::T

    h::T

    A::M
    B::M
    e::V

    Ad::M
    Bd::M
    ed::V

    Δt::T

    buffer::Matrix{Int}

    end_layer::Bool

    function PBFPowerFieldCylindrical(nz, nr, Δr, Δz, k, ρ, cₚ, T∞, h, Δt, T, V, M; buffer=[0 0], end_layer=false)

        A, B, e = matrices_for_voxel_conduction(nz, nr, Δr, Δz, k, ρ, cₚ, h, T∞, buffer)

        Ad, Bd, ed = discretize_linear_dynamics(A, B, e, Δt)
        # Ad, Bd, ed = discretize_linear_dynamics_approximation(A, B, e, Δt)

        if end_layer
            A_shuffle = ones(nz+1, nr)
            A_shuffle[1, :] .= 0
            A_shuffle = diagm([vec(A_shuffle); 1; 1])
            indices = vec(any(A_shuffle .!= 0, dims=1))
            A_shuffle = M(A_shuffle[:, indices])

            B_shuffle = zeros(nz+1, nr)
            B_shuffle = diagm([vec(B_shuffle); 1; 1])
            B_shuffle = M(B_shuffle[:, indices])

            e_shuffle = ones(nz+1, nr)
            e_shuffle[1, :] .= 0
            e_shuffle = diagm([vec(e_shuffle); 1; 1])
            e_shuffle = M(e_shuffle[:, indices])

            Ad = A_shuffle * Ad
            Bd = B_shuffle * Bd
            ed = e_shuffle * ed
            ed[1:(nz+1):(end-2)] .= T∞
        end

        new{T,V,M}(nz, nr, Δr, Δz, k, ρ, cₚ, T∞, h, A, B, e, Ad, Bd, ed, Δt, buffer, end_layer)
    end
end

nx(dynamics::PBFPowerFieldCylindrical)::Int = dynamics.nz * dynamics.nr + 2
nu(dynamics::PBFPowerFieldCylindrical)::Int = 2 ## Hard coded ## 
Δt(dynamics::PBFPowerFieldCylindrical, u) = dynamics.Δt

function rate!(dynamics::PBFPowerFieldCylindrical{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    A, B = dynamics.A, dynamics.B

    mul!(ẋ, A, x, 1.0, 1.0)
    mul!(ẋ, B, u, 1.0, 1.0)
end

function transition!(dynamics::PBFPowerFieldCylindrical, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    Ad, Bd, ed = dynamics.Ad, dynamics.Bd, dynamics.ed

    xₖ₊₁ .= ed
    mul!(xₖ₊₁, Ad, xₖ, 1.0, 1.0)
    mul!(xₖ₊₁, Bd, uₖ, 1.0, 1.0)
end

function transition_state_jacobian!(dynamics::PBFPowerFieldCylindrical, A, xₖ, uₖ)
    A .= dynamics.Ad
end

function transition_input_jacobian!(dynamics::PBFPowerFieldCylindrical, B, xₖ, uₖ)
    B .= dynamics.Bd
end