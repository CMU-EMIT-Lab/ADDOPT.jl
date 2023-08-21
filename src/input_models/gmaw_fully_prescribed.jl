using StatsFuns
using Interpolations
using Symbolics

struct GMAWDynamicsFullyPrescribed <: InputDynamics
    nx::Int
    ny::Int
    nz::Int

    l::Float64

    xₙ::Vector{Float64}
    yₙ::Vector{Float64}
    zₙ::Vector{Float64}

    ρ::Float64
    cₚ::Float64
    T∞::Float64
    Tₗ::Float64
    wire_diam::Float64

    h∞::Float64
    h₀::Float64
    hₐᵣ::Float64

    η::Float64
    rₚ::Float64
    Δr::Vector{Float64}

    p̄::Vector{Vector{Float64}}

    x::Vector{Vector{Float64}}
    xcur::Vector{Float64}

    Fp_cache::Dict{Tuple{DataType,Int},Any}
    Fm_cache::Dict{Tuple{DataType,Int},Any}

    function GMAWDynamicsFullyPrescribed(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, rₚ, Δr, p̄, x)
        Fp = Dict{Tuple{DataType,Int},Any}()
        Fm = Dict{Tuple{DataType,Int},Any}()

        xc = zeros(size(x[1]))

        return new(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, rₚ, Δr, p̄, x, xc, Fp, Fm)
    end
end

Nu(id::GMAWDynamicsFullyPrescribed)::Int = 1 # trim
Nr(id::GMAWDynamicsFullyPrescribed)::Int = 0

input_min(id::GMAWDynamicsFullyPrescribed) = [0.9] # trim
input_max(id::GMAWDynamicsFullyPrescribed) = [1.1]
input_idle(id::GMAWDynamicsFullyPrescribed) = [0.0]

state_min(id::GMAWDynamicsFullyPrescribed) = []
state_max(id::GMAWDynamicsFullyPrescribed) = []

function dynamics_function!(id::GMAWDynamicsFullyPrescribed, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}

end

function input_function!(id::GMAWDynamicsFullyPrescribed, ds::AbstractVector{Ty}, r, u, t, zi, Δt) where {Ty}
    nz, ny, nx = id.nz, id.ny, id.nx
    N = nx * ny * nz
    dE = view(ds, 1:N)
    trim = u[1]
    l, xₙ, yₙ, zₙ = id.l, id.xₙ, id.yₙ, id.zₙ
    ρ, cₚ, T∞, wire_diam = id.ρ, id.cₚ, id.T∞, id.wire_diam
    h∞, h₀, hₐᵣ, η, rₚ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.rₚ
    xar = id.x
    x = id.xcur
    
    zint = floor(Int, zi)
    Δr = id.Δr[zint]
    WFS = 2Δr / Δt * (rₚ / wire_diam)^2

    if zint == zi
        x .= xar[zint]
    else
        @. x = (1 - (zi - zint)) * xar[zint] + (zi - zint) * xar[zint+1]
        Δr = (1 - (zi - zint)) * id.Δr[zint] + (zi - zint) * id.Δr[zint+1]
    end

    if eltype(ds) == Symbolics.Num
        x .= 1.0
    end

    P = (η * 29000 + cₚ * ρ * (wire_diam/2.0)^2 * π * T∞) * WFS

    # Forced / input dynamics
    dE .+= x .* (trim * P)              # Add in torch power
end