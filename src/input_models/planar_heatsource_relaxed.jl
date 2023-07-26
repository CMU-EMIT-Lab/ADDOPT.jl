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

    vₘₐₓ::Float64

    Qx::Matrix{Float64}
    Qz::Matrix{Float64}
    Qc::Matrix{Float64}

    P2_cache::Dict{DataType,Any}

    function PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, ρ, cₚ, σ, vₘₐₓ)
        nvox = nrows * ncols

        Qx = ones(nvox) * (xₙ .^ 2)' - (xₙ * xₙ') - σ^2 * ones(nvox, nvox)
        Qz = ones(nvox) * (zₙ .^ 2)' - (zₙ * zₙ') - σ^2 * ones(nvox, nvox)
        Qc = ones(nvox) * (xₙ .* zₙ)' - (xₙ * zₙ')

        P2 = Dict{DataType,Any}()

        return new(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, σ, ρ, cₚ, vₘₐₓ, 20Qx, 20Qz, 20Qc, P2)
    end
end

Nu(id::PlanarHeatsourceDynamics)::Int = id.nrows * id.ncols + 2 # P (W for power), vx, vz (m/s)
Nr(id::PlanarHeatsourceDynamics)::Int = 2 # x, z position (m)

Nc_ineq(id::PlanarHeatsourceDynamics)::Int = 2 #3 # Power sum and variance
Nc_eq(id::PlanarHeatsourceDynamics)::Int = 2 #3 # COM constraints, 0 covariance

ineq_min(id::PlanarHeatsourceDynamics) = [id.Pₘᵢₙ; id.Pₘᵢₙ^2]#-Inf; -Inf]
ineq_max(id::PlanarHeatsourceDynamics) = [id.Pₘₐₓ; id.Pₘₐₓ^2]#0.0; 0.0]

input_min(id::PlanarHeatsourceDynamics) = [zeros(id.nrows * id.ncols); -id.vₘₐₓ; -id.vₘₐₓ]
input_max(id::PlanarHeatsourceDynamics) = [Inf * ones(id.nrows * id.ncols); id.vₘₐₓ; id.vₘₐₓ]
input_idle(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols + 2)

state_min(id::PlanarHeatsourceDynamics) = [0.0; 0.0]
state_max(id::PlanarHeatsourceDynamics) = [id.l * (id.ncols+1); id.l * (id.nrows+1)]

function dynamics_function!(id::PlanarHeatsourceDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}
    nvox = id.nrows * id.ncols
    vx = u[nvox+1]
    vz = u[nvox+2]

    dr[1] = vx
    dr[2] = vz
end

function input_function!(id::PlanarHeatsourceDynamics, ds::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    dT = ds
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    P = view(u, 1:nvox)
    ρ, cₚ = id.ρ, id.cₚ

    # Forced / input dynamics
    @. dT += P / (ρ * l^3 * cₚ)
end

function equality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    P = view(u, 1:nvox)
    xₜ = r[1]
    zₜ = r[2]

    c[1] = dot(xₙ, P) - xₜ * sum(P)
    c[2] = dot(zₙ, P) - zₜ * sum(P)
    # c[3] = 0.5 * dot(P, id.Qc, P)
end

function inequality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    P = view(u, 1:nvox)
    Qx, Qz = id.Qx, id.Qz
    Q = Diagonal(ones(nvox))

    P2 = get!(id.P2_cache, Ty) do
        zeros(Ty, nvox)
    end::Vector{Ty}

    c[1] = sum(P) # Max and min power constraint
    # c[2] = 0.5 * dot(P, Qx, P) # X power variance
    # c[3] = 0.5 * dot(P, Qz, P) # Z power variance

    P2 .= P
    P2 .^= 2
    c[2] = sum(P2)
end