using StatsFuns

struct PlanarGMAWDynamicsPrescribed <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    k
    ρ::Float64
    cₚ::Float64
    T∞::Float64
    Tₗ::Float64
    wire_diam::Float64

    h∞::Float64
    h₀::Float64
    hₐᵣ::Float64
    η::Float64
    γᵣ::Float64
    γₕ::Float64
    wₓ::Float64
    bₕ::Float64

    xmin::Float64
    xmax::Float64
    tmin::Float64
    tmax::Float64
    
    F_cache::Dict{DataType,Any}
    ZW_cache::Dict{DataType,Any}

    function PlanarGMAWDynamicsPrescribed(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, xmin, xmax, tmin, tmax)
        F = Dict{DataType,Any}()
        ZW = Dict{DataType,Any}()

        return new(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ,  xmin, xmax, tmin, tmax, F, ZW)
    end
end

Nu(id::PlanarGMAWDynamicsPrescribed)::Int = 1 # WFS (m/s for speeds)
Nr(id::PlanarGMAWDynamicsPrescribed)::Int = 1 # meltpool radius

input_min(id::PlanarGMAWDynamicsPrescribed) = [0.001] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::PlanarGMAWDynamicsPrescribed) = [0.1]
input_idle(id::PlanarGMAWDynamicsPrescribed) = [0.0]

state_min(id::PlanarGMAWDynamicsPrescribed) = [id.l]
state_max(id::PlanarGMAWDynamicsPrescribed) = [id.l*10]

function dynamics_function!(id::PlanarGMAWDynamicsPrescribed, dr::AbstractVector{Ty}, s, r, u, t) where {Ty}
    N = length(s) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    rₘₚ = r[1]
    WFS = u[1]
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, Tₗ, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.Tₗ, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ
    xmin, xmax, tmin, tmax = id.xmin, id.xmax, id.tmin, id.tmax
    TS = abs(xmax - xmin) / (tmax - tmin)

    r̄ₘₚ = wire_diam * √((WFS + .001) / (TS + .001)) * √(1 / 2) #: wire_diam # (eltype(ZW) == Symbolics.Num) || (WFS > 0 && TS > 0) ? 

    dr[1] = γᵣ * (r̄ₘₚ - rₘₚ)
end

function input_function!(id::PlanarGMAWDynamicsPrescribed, ds::AbstractVector{Ty}, r, u, t) where {Ty}
    N = length(ds) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    rₘₚ = r[1]
    WFS = u[1]
    # xₜ, zₜ, rₘₚ, zₘₚ = r[1], r[2], r[3], r[4]    
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ
    xmin, xmax, tmin, tmax = id.xmin, id.xmax, id.tmin, id.tmax
    TS = abs(xmax - xmin) / (tmax - tmin)
    zₘₚ = l
    xₜ = clamp((xmax - xmin) / (tmax - tmin) * (t - tmin) + xmin, xmin, xmax)

    F = get!(id.F_cache, Ty) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    ṁ = WFS * π * (wire_diam / 2)^2 * ρ # kg/s
    P = 29000 * WFS # W/(m/s) * (m/s)

    @. F = (l^2 / (2π*(l/wₓ)*rₘₚ)) * exp(-((xₙ - xₜ)^2/(l/wₓ)^2 + (zₙ - zₘₚ)^2/rₘₚ^2)/2)
    # @. F = normpdf((xₙ - xₜ) / l * wₓ) * normpdf((zₙ - zₘₚ) / rₘₚ) #* logistic((zₙ - zₘₚ) / l + bₕ) 

    # if eltype(F) == Symbolics.Num
    #     @. F += xₜ + rₘₚ + zₘₚ + WFS
    # else
    #     F ./= sum(F)
    # end 

    # Forced / input dynamics
    @. dE += η * F * P              # Add in torch power
    @. dE += F * (cₚ * ṁ * T∞)      # Add in energy contribution from incoming wire (assume room temp)
    @. dm += F * ṁ
end