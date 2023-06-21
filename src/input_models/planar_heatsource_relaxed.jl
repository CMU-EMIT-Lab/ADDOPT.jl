using LinearAlgebra

struct PlanarHeatsourceDynamics <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    Pₘₐₓ::Float64
    Pₘᵢₙ::Float64

    σ::Float64

    ρ::Float64
    cₚ::Float64

    Qx::Matrix{Float64}
    Qz::Matrix{Float64}

    function PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, ρ, cₚ, σ)
        nvox = nrows*ncols

        Qx = ones(nvox) * (xₙ.^2)' - (xₙ*xₙ') - σ^2 * ones(nvox, nvox)
        Qz = ones(nvox) * (zₙ.^2)' - (zₙ*zₙ') - σ^2 * ones(nvox, nvox)

        return new(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, σ, ρ, cₚ, Qx, Qz)
    end
end

Nu(id::PlanarHeatsourceDynamics)::Int = id.nrows * id.ncols # vx, vz, P (m/s for speeds, W for power), slacks
Nr(id::PlanarHeatsourceDynamics)::Int = 0#2 # torch position (x,z)

Nc_ineq(id::PlanarHeatsourceDynamics)::Int = 3#id.nrows * id.ncols + 1
Nc_eq(id::PlanarHeatsourceDynamics)::Int = 0#1#id.nrows * id.ncols

ineq_min(id::PlanarHeatsourceDynamics) = [0.0; -Inf; -Inf]#[zeros(Nc_ineq(id))]
ineq_max(id::PlanarHeatsourceDynamics) = [id.Pₘₐₓ; 0.0; 0.0]#Inf * ones(Nc_ineq(id))

input_min(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)#03; -0.03; id.Pₘᵢₙ; zeros(id.nrows * id.ncols)] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::PlanarHeatsourceDynamics) = Inf * ones(id.nrows * id.ncols)#[0.03; 0.03; id.Pₘₐₓ; Inf * ones(id.nrows * id.ncols)] # fourth input is std slack
input_idle(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)#[0.0; 0.0; 0.0; zeros(id.nrows * id.ncols)]

state_min(id::PlanarHeatsourceDynamics) = []#[0.0; 0.0]
state_max(id::PlanarHeatsourceDynamics) = []#[(id.ncols + 1) * id.l; (id.nrows + 1) * id.l]

function dynamics_function!(id::PlanarHeatsourceDynamics, dr::AbstractVector{Ty}, s, r, u, t) where {Ty}
    # xₜ, zₜ = r[1], r[2]
    # vx, vz, P = u[1], u[2], u[3]

    # dr[1] = vx
    # dr[2] = vz
end

function input_function!(id::PlanarHeatsourceDynamics, ds::AbstractVector{Ty}, r, u, t) where {Ty}
    dT = ds
    # xₜ, zₜ = r[1], r[2]
    # vx, vz, P = u[1], u[2], u[3]
    # λ = @view u[4:end]
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    ρ, cₚ = id.ρ, id.cₚ

    # Forced / input dynamics
    # @. dT += ((P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) + λ) / (ρ * l^3 * cₚ)  # Add in torch power
    @. dT += u / (ρ * l^3 * cₚ)
end

function equality_constraint!(id::PlanarHeatsourceDynamics, c, r, u, t)
    # xₜ, zₜ = r[1], r[2]
    # vx, vz, P = u[1], u[2], u[3]
    # λ = @view u[4:end]
    # l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ


    # c[1] = sum(@. (P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) + λ) - P
    # @. c = ((P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) + λ) * λ
end

function inequality_constraint!(id::PlanarHeatsourceDynamics, c, r, u, t)
    # xₜ, zₜ = r[1], r[2]
    # vx, vz, P = u[1], u[2], u[3]
    # λ = @view u[4:end]
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    Qx, Qz = id.Qx, id.Qz

    c[1] = sum(u)
    c[2] = dot(u, Qx, u)
    c[3] = dot(u, Qz, u)
    # @. c[1:nvox] = (P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) + λ
    # c[nvox+1] = P - sum(@. (P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) + λ)
    # @. c[(nvox+1):2nvox] = (P * (2l^2 / (π * σ^4)) * (σ^2 - (xₙ - xₜ)^2 - (zₙ - zₜ)^2 - l^2 / 6)) * -λ
end