export FurnaceTempering
struct FurnaceTempering{T} <: Dynamics{T}
    furnace::Furnace{T}
    tempering::Tempering{T}

    function FurnaceTempering(furnace::Furnace{T}, tempering::Tempering{T}) where {T}
        @assert nu(tempering) == nx(furnace)
        @assert tempering.Δt == furnace.Δt

        new{T}(furnace, tempering)
    end
end

nx(dynamics::FurnaceTempering)::Int = nx(dynamics.furnace) + nx(dynamics.tempering)
nu(dynamics::FurnaceTempering)::Int = nu(dynamics.furnace)
Δt(dynamics::FurnaceTempering, u) = Δt(dynamics.furnace, u)

function rate!(dynamics::FurnaceTempering, ẋ::AbstractVector{E}, x, u) where {E}
    furnace, tempering = dynamics.furnace, dynamics.tempering
    T = @view x[1:nx(furnace)]
    ŷ = @view x[(nx(furnace)+1):end]
    P = u

    dT = @view ẋ[1:nx(furnace)]
    dŷ = @view ẋ[(nx(furnace)+1):end]
    rate!(furnace, dT, T, P)
    rate!(tempering, dŷ, ŷ, T)
end

function transition!(dynamics::FurnaceTempering, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    furnace, tempering = dynamics.furnace, dynamics.tempering
    Tₖ = @view xₖ[1:nx(furnace)]
    ŷₖ = @view xₖ[(nx(furnace)+1):end]
    P = uₖ

    Tₖ₊₁ = @view xₖ₊₁[1:nx(furnace)]
    ŷₖ₊₁ = @view xₖ₊₁[(nx(furnace)+1):end]
    transition!(furnace, Tₖ₊₁, Tₖ, P)
    transition!(tempering, ŷₖ₊₁, ŷₖ, Tₖ)
end

function transition_state_jacobian!(dynamics::FurnaceTempering, A, xₖ, uₖ)
    furnace, tempering = dynamics.furnace, dynamics.tempering
    nf = nx(furnace)
    Tₖ = @view xₖ[1:nf]
    ŷₖ = @view xₖ[(nf+1):end]

    A .= 0.0
    transition_state_jacobian!(furnace, (@view A[1:nf, 1:nf]), Tₖ, uₖ)
    transition_input_jacobian!(tempering, (@view A[(nf+1):end, 1:nf]), ŷₖ, Tₖ)
    transition_state_jacobian!(tempering, (@view A[(nf+1):end, (nf+1):end]), ŷₖ, Tₖ)
end

function transition_input_jacobian!(dynamics::FurnaceTempering, B, xₖ, uₖ)
    furnace, tempering = dynamics.furnace, dynamics.tempering
    nf = nx(furnace)
    Tₖ = @view xₖ[1:nf]

    B .= 0.0
    transition_input_jacobian!(furnace, (@view B[1:nf, :]), Tₖ, uₖ)
end

function transition_state_jacobian_product_state_jacobian!(dynamics::FurnaceTempering, ∂Ap∂x, p, xₖ, uₖ)
    furnace, tempering = dynamics.furnace, dynamics.tempering
    nf = nx(furnace)
    Tₖ = @view xₖ[1:nf]
    ŷₖ = @view xₖ[(nf+1):end]
    py = @view p[(nf+1):end]
    Ap11 = @view ∂Ap∂x[1:nf, 1:nf]
    Ap12 = @view ∂Ap∂x[1:nf, (nf+1):end]
    Ap21 = @view ∂Ap∂x[(nf+1):end, 1:nf]
    Ap22 = @view ∂Ap∂x[(nf+1):end, (nf+1):end]

    transition_input_jacobian_product_input_jacobian!(tempering, Ap11, py, ŷₖ, Tₖ)
    transition_input_jacobian_product_state_jacobian!(tempering, Ap12, py, ŷₖ, Tₖ)
    transition_input_jacobian_product_state_jacobian!(tempering, Ap21, py, ŷₖ, Tₖ)
    transition_state_jacobian_product_state_jacobian!(tempering, Ap22, py, ŷₖ, Tₖ)
end

function transition_input_jacobian_product_input_jacobian!(dynamics::FurnaceTempering, ∂Bp∂u, p, xₖ, uₖ)
    ∂Bp∂u .= 0
end

function transition_input_jacobian_product_state_jacobian!(dynamics::FurnaceTempering, ∂Bp∂x, p, xₖ, uₖ)
    ∂Bp∂x .= 0
end