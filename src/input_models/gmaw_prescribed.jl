using StatsFuns
using Interpolations
using Symbolics

struct GMAWDynamicsPrescribed <: InputDynamics
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
    rₘ::Float64

    x̄::Interpolations.Extrapolation
    ȳ::Interpolations.Extrapolation
    z̄::Interpolations.Extrapolation
    t̄::Vector{Float64}

    Fp_cache::Dict{DataType,Any}
    Fm_cache::Dict{DataType,Any}

    function GMAWDynamicsPrescribed(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, rₚ, rₘ, p̄, t̄)
        Fp = Dict{DataType,Any}()
        Fm = Dict{DataType,Any}()

        x̄ = linear_interpolation(t̄, [p[1] for p in p̄], extrapolation_bc=Flat())
        ȳ = linear_interpolation(t̄, [p[2] for p in p̄], extrapolation_bc=Flat())
        z̄ = linear_interpolation(t̄, [p[3] for p in p̄], extrapolation_bc=Flat())

        return new(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, rₚ, rₘ, x̄, ȳ, z̄, t̄, Fp, Fm)
    end
end

Nu(id::GMAWDynamicsPrescribed)::Int = 1 # WFS (m/s for speeds)
Nr(id::GMAWDynamicsPrescribed)::Int = 0

input_min(id::GMAWDynamicsPrescribed) = [0.030] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::GMAWDynamicsPrescribed) = [0.300]
input_idle(id::GMAWDynamicsPrescribed) = [0.0]

state_min(id::GMAWDynamicsPrescribed) = []
state_max(id::GMAWDynamicsPrescribed) = []

function dynamics_function!(id::GMAWDynamicsPrescribed, dr::AbstractVector{Ty}, s, r, u, t) where {Ty}

end

function input_function!(id::GMAWDynamicsPrescribed, ds::AbstractVector{Ty}, r, u, t) where {Ty}
    N = length(ds) ÷ 2
    dE = view(ds, 1:N)
    dx = view(ds, (N+1):2N)
    nz, ny, nx = id.nz, id.ny, id.nx
    WFS = u[1]
    l, xₙ, yₙ, zₙ = id.l, id.xₙ, id.yₙ, id.zₙ
    ρ, cₚ, T∞, wire_diam = id.ρ, id.cₚ, id.T∞, id.wire_diam
    h∞, h₀, hₐᵣ, η, rₚ, rₘ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.rₚ, id.rₘ

    xₜ::Float64 = typeof(t) == Symbolics.Num ? l : id.x̄(t)
    yₜ::Float64 = typeof(t) == Symbolics.Num ? l : id.ȳ(t)
    zₜ::Float64 = typeof(t) == Symbolics.Num ? l : id.z̄(t)

    Fp = get!(id.Fp_cache, Ty) do
        zeros(Ty, nx * ny * nz)
    end::Vector{Ty}

    Fm = get!(id.Fm_cache, Ty) do
        zeros(Ty, nx * ny * nz)
    end::Vector{Ty}

    ṁ = WFS * π * (wire_diam / 2)^2 * ρ # kg/s
    P = 29000 * WFS # W/(m/s) * (m/s)

    Fp .= (l^3 / ((2π)^(3 / 2) * rₚ^3)) .* exp.(((xₙ .- xₜ) .* (xₙ .- xₜ) .+ (yₙ .- yₜ) .* (yₙ .- yₜ) .+ (zₙ .- zₜ) .* (zₙ .- zₜ)) ./ (-2rₚ^2))
    Fm .= (l^3 / ((2π)^(3 / 2) * rₘ^3)) .* exp.(((xₙ .- xₜ) .* (xₙ .- xₜ) .+ (yₙ .- yₜ) .* (yₙ .- yₜ) .+ (zₙ .- zₜ) .* (zₙ .- zₜ)) ./ (-2rₘ^2))

    # Forced / input dynamics
    dE .+= Fp .* (η * P)              # Add in torch power
    dE .+= Fm .* (cₚ * ṁ * T∞)      # Add in energy contribution from incoming wire (assume room temp)
    dx .+= Fm .* (ṁ / (l^3 * ρ))
end

function interp(t, ts, ps)
    idx = searchsortedfirst(ts, t)

    if idx == 1
        return ps[1]
    elseif idx == length(ts) + 1
        return ps[end]
    end

    frac = 1 - (ts[idx] - t) / (ts[idx] - ts[idx-1])

    return @. ps[idx-1] + frac * (ps[idx] - ps[idx-1])
end