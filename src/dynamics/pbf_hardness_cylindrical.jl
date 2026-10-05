export PBFTempering_cylindrical
struct PBFTempering_cylindrical{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    pbf_powerfield::PBFPowerFieldCylindrical{T,V,M}
    tempering::Tempering{T}
end

nx(dynamics::PBFTempering_cylindrical)::Int = nx(dynamics.pbf_powerfield) + nx(dynamics.tempering)
nxₖ₊₁(dynamics::PBFTempering_cylindrical)::Int = nx(dynamics)
nu(dynamics::PBFTempering_cylindrical)::Int = nu(dynamics.pbf_powerfield)
Δt(dynamics::PBFTempering_cylindrical, u) = Δt(dynamics.pbf_powerfield, u)

function rate!(dynamics::PBFTempering_cylindrical, ẋ, x, u)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    Nu = 2

    T = @view x[1:((length(x)-Nu)÷2+Nu)]
    ŷ = @view x[((length(x)-Nu)÷2+1):(end-Nu)]
    P = u

    dT = @view ẋ[1:((length(ẋ)-Nu)÷2+Nu)]
    dŷ = @view ẋ[((length(ẋ)-Nu)÷2+1):(end-Nu)]
    rate!(pbf_powerfield, dT, T, P)
    rate!(tempering, dŷ, ŷ, T[1:(end-Nu)])
end

function transition!(dynamics::PBFTempering_cylindrical, xₖ₊₁, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    nz, nr = pbf_powerfield.nz, pbf_powerfield.nr
    Nu = 2

    Tₖ = @view xₖ[1:((length(xₖ)-Nu)÷2+Nu)]
    ŷₖ = @view xₖ[((length(xₖ)-Nu)÷2+Nu+1):end]
    P = uₖ

    Tₖ₊₁ = @view xₖ₊₁[1:((length(xₖ₊₁)-Nu)÷2+Nu)]
    ŷₖ₊₁ = @view xₖ₊₁[((length(xₖ₊₁)-Nu)÷2+Nu+1):end]

    transition!(pbf_powerfield, Tₖ₊₁, Tₖ, P)


    if pbf_powerfield.end_layer != (length(ŷₖ₊₁) > length(ŷₖ))
        println("What!")
    end

    if length(ŷₖ₊₁) > length(ŷₖ)
        # This involves copying and allocation unfortunately due to the choice to expand on rows and prepend
        # @jamccauley3 consider changing this (would require flipping the convention in a lot of places)
        # -- @mkhrenov (7/11/2026)
        ŷ_mat = reshape(ŷₖ₊₁, :, nr)
        ŷ_mat_prev = ŷ_mat[2:end, :]
        ŷ_vec_prev = vec(ŷ_mat_prev)

        transition!(tempering, ŷ_vec_prev, ŷₖ, Tₖ[1:(end-Nu)])

        ŷ_mat[1, :] .= tempering.y_init
        ŷ_mat[2:end, :] .= ŷ_mat_prev
    else
        transition!(tempering, ŷₖ₊₁, ŷₖ, Tₖ[1:(end-Nu)])
    end
end

function transition_state_jacobian!(dynamics::PBFTempering_cylindrical, A, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    Nu = 2
    nT = nx(pbf_powerfield) - Nu
    Nx = nx(pbf_powerfield)
    Tₖ = @view xₖ[1:nT]
    ŷₖ = @view xₖ[(Nx+1):end]

    A .= 0.0

    if pbf_powerfield.end_layer
        nr = pbf_powerfield.nr
        transition_state_jacobian!(pbf_powerfield, (@view A[1:(Nx+nr), 1:Nx]), view(xₖ, 1:Nx), uₖ)
        transition_input_jacobian!(tempering, (@view A[(Nx+nr+1):end, 1:nT]), ŷₖ, Tₖ)
        transition_state_jacobian!(tempering, (@view A[(Nx+nr+1):end, (Nx+1):end]), ŷₖ, Tₖ)
    else
        transition_state_jacobian!(pbf_powerfield, (@view A[1:Nx, 1:Nx]), view(xₖ, 1:Nx), uₖ)
        transition_input_jacobian!(tempering, (@view A[(Nx+1):end, 1:nT]), ŷₖ, Tₖ)
        transition_state_jacobian!(tempering, (@view A[(Nx+1):end, (Nx+1):end]), ŷₖ, Tₖ)
    end
end

function transition_input_jacobian!(dynamics::PBFTempering_cylindrical, B, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    Nx = nx(pbf_powerfield)
    Tₖ = @view xₖ[1:Nx]

    B .= 0.0
    if pbf_powerfield.end_layer
        nr = pbf_powerfield.nr
        transition_input_jacobian!(pbf_powerfield, (@view B[1:(Nx+nr), :]), Tₖ, uₖ)
    else
        transition_input_jacobian!(pbf_powerfield, (@view B[1:Nx, :]), Tₖ, uₖ)
    end
end

function transition_state_jacobian_product_state_jacobian!(dynamics::PBFTempering_cylindrical, ∂Ap∂x, p, xₖ, uₖ)
    pbf_powerfield, tempering = dynamics.pbf_powerfield, dynamics.tempering
    Nu = nu(pbf_powerfield)
    nT = nx(pbf_powerfield) - Nu
    Tₖ = @view xₖ[1:nT]
    ŷₖ = @view xₖ[(nT+Nu+1):end]
    py = @view p[(end-nT+1):end]
    Ap11 = @view ∂Ap∂x[1:nT, 1:nT]
    Ap12 = @view ∂Ap∂x[1:nT, (nT+Nu+1):end]
    Ap21 = @view ∂Ap∂x[(nT+Nu+1):end, 1:nT]
    Ap22 = @view ∂Ap∂x[(nT+Nu+1):end, (nT+Nu+1):end]

    ∂Ap∂x .= 0.0

    transition_input_jacobian_product_input_jacobian!(tempering, Ap11, py, ŷₖ, Tₖ)
    transition_input_jacobian_product_state_jacobian!(tempering, Ap12, py, ŷₖ, Tₖ)
    transition_input_jacobian_product_state_jacobian!(tempering, Ap21, py, ŷₖ, Tₖ)
    transition_state_jacobian_product_state_jacobian!(tempering, Ap22, py, ŷₖ, Tₖ)
end

function transition_input_jacobian_product_input_jacobian!(dynamics::PBFTempering_cylindrical, ∂Bp∂u, p, xₖ, uₖ)
    ∂Bp∂u .= 0
end

function transition_input_jacobian_product_state_jacobian!(dynamics::PBFTempering_cylindrical, ∂Bp∂x, p, xₖ, uₖ)
    ∂Bp∂x .= 0
end