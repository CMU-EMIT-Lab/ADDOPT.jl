using LinearAlgebra

struct PlanarHeatsourceDynamics <: InputDynamics
    nrows::Int
    ncols::Int
    ndeep::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    Pₘₐₓ::Float64
    Pₘᵢₙ::Float64

    σ::Float64

    ρ::Float64
    cₚ::Float64

    vₘₐₓ::Float64
    Plim::Vector{Float64}

    function PlanarHeatsourceDynamics(nrows, ncols, ndeep, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, ρ, cₚ, σ, vₘₐₓ, Plim)
        return new(nrows, ncols, ndeep, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, σ, ρ, cₚ, vₘₐₓ, Plim)
    end
end

Nu(id::PlanarHeatsourceDynamics)::Int = id.nrows * id.ncols
Nr(id::PlanarHeatsourceDynamics)::Int = 0

Nc_ineq(id::PlanarHeatsourceDynamics)::Int = 1 # Power sum
Nc_eq(id::PlanarHeatsourceDynamics)::Int = 0

ineq_min(id::PlanarHeatsourceDynamics) = [id.Pₘᵢₙ]
ineq_max(id::PlanarHeatsourceDynamics) = [id.Pₘₐₓ]

input_min(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)
input_max(id::PlanarHeatsourceDynamics) = id.Plim
input_idle(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)

state_min(id::PlanarHeatsourceDynamics) = []
state_max(id::PlanarHeatsourceDynamics) = []

function dynamics_function!(id::PlanarHeatsourceDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}

end

function input_function!(id::PlanarHeatsourceDynamics, ds::AbstractVector{Ty}, r, u, t, zi, Δt) where {Ty}
    dT = ds
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    P = view(u, 1:nvox)
    ρ, cₚ = id.ρ, id.cₚ
    C = ρ * cₚ * l^3 
    
    top_dT = view(dT, 1:nvox)

    # Forced / input dynamics
    @. top_dT += P / C
end

function equality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}

end

function inequality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    P = view(u, 1:nvox)

    c[1] = sum(P) # Max and min power constraint
end