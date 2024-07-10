
struct PBFTempering{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    pbf_powerfield::PBFPowerField{T,V,M}
    tempering::Tempering{T}
end

nx(dynamics::PBFTempering)::Int = nx(dynamics.pbf_powerfield) + nx(dynamics.tempering)
nxₖ₊₁(dynamics::PBFTempering)::Int = nx(dynamics)
nu(dynamics::PBFTempering)::Int = nu(dynamics.pbf_powerfield)
Δt(dynamics::PBFTempering, u) = Δt(dynamics.pbf_powerfield, u)

function rate!(dynamics::PBFTempering, ẋ, x, u)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    T = @view x[1:nx(pbf_powerfield)]
    ŷ = @view x[(nx(pbf_powerfield)+1):end]
    P = u

    dT = @view ẋ[1:nx(pbf_powerfield)]
    dŷ = @view ẋ[(nx(pbf_powerfield)+1):end]
    rate!(pbf_powerfield, dT, T, P)
    rate!(tempering, dŷ, ŷ, T)
end

function transition!(dynamics::PBFTempering, xₖ₊₁, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    Tₖ = @view xₖ[1:nx(pbf_powerfield)]
    ŷₖ = @view xₖ[(nx(pbf_powerfield)+1):end]
    P = uₖ

    Tₖ₊₁ = @view xₖ₊₁[1:nx(pbf_powerfield)]
    ŷₖ₊₁ = @view xₖ₊₁[(nx(pbf_powerfield)+1):end]
    transition!(pbf_powerfield, Tₖ₊₁, Tₖ, P)
    transition!(tempering, ŷₖ₊₁, ŷₖ, Tₖ)
end

function transition_state_jacobian!(dynamics::PBFTempering, A, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    nT = nx(pbf_powerfield)
    Tₖ = @view xₖ[1:nT]
    ŷₖ = @view xₖ[(nT+1):end]
    
    A .= 0.0
    transition_state_jacobian!(pbf_powerfield, (@view A[1:nT, 1:nT]), Tₖ, uₖ)
    transition_input_jacobian!(tempering, (@view A[(nT+1):end, 1:nT]), ŷₖ, Tₖ)
    transition_state_jacobian!(tempering, (@view A[(nT+1):end, (nT+1):end]), ŷₖ, Tₖ)
end

function transition_input_jacobian!(dynamics::PBFTempering, B, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    nT = nx(pbf_powerfield)
    Tₖ = @view xₖ[1:nT]
    
    B .= 0.0
    transition_input_jacobian!(pbf_powerfield, (@view B[1:nT, :]), Tₖ, uₖ)
end