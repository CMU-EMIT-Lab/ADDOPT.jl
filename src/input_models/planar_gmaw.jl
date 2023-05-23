struct PlanarGMAWDynamics <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    k
    ρ::Float64
    cₚ::Float64
    T∞::Float64
    wire_diam::Float64

    h∞::Float64
    h₀::Float64
    hₐᵣ::Float64
    η::Float64
    γᵣ::Float64
    γₕ::Float64
    wₓ::Float64
    bₕ::Float64

    F_cache::Dict{DataType, Any}
    ZW_cache::Dict{DataType, Any}

    function PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
        F = Dict{DataType, Any}()
        ZW = Dict{DataType, Any}()

        return new(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, F, ZW)
    end
end

Nu(id::PlanarGMAWDynamics) = 4 # vx, vz, trim, WFS (m/s for speeds)
Nr(id::PlanarGMAWDynamics) = 4 # torch position (x,z), meltpool radius, meltpool root (z) (m)

input_min(id::PlanarGMAWDynamics) = [-Inf; -Inf; 0.5; 0.001] # vx, vz, trim, WFS (m/s for speeds)
input_max(id::PlanarGMAWDynamics) = [Inf; Inf; 1.2; 0.085]
input_idle(id::PlanarGMAWDynamics) = [0.0; 0.0; 0.0; 0.0]

state_min(id::PlanarGMAWDynamics) = [-Inf; -Inf; 0.0; 0.0]
state_max(id::PlanarGMAWDynamics) = [Inf; Inf; Inf; Inf]

function dynamics_function!(id::PlanarGMAWDynamics, dr::AbstractVector{Ty}, s, r, u) where Ty
    N = length(s) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ, rₘₚ, zₘₚ = r[1], r[2], r[3], r[4]
    vx, vz, trim, WFS = u[1], u[2], u[3], u[4]
    TS = sqrt(vx^2 + vz^2)
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, T₀, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.T₀, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

    ZW = get!(id.ZW_cache, Ty) do
        zeros(Ty, n_rows*n_cols)
    end::Vector{Ty}
    
    z̄ₘₚ = zₘₚ
    # Compute steady state meltpool radius and z location
    r̄ₘₚ = WFS > 0 && TS > 0 ? wire_diam * √(WFS / TS) * √(1 / 2) : wire_diam
    @. ZW = normpdf((xₙ - xₜ) / l * 2) * (1 - exp(-m / (ρ * l^2))) * max(sign(Tₗ * m * cₚ - E), 0) * exp((zₙ / l) * 10)
    ZW_sum = sum(ZW) # logistic(1000*(Tₗ*m*cₚ - E) - 20)
    if ZW_sum > 0 && WFS > 0 && TS > 0
        ZW .*= zₙ
        z̄ₘₚ = sum(ZW) / ZW_sum + l
    end

    dr[1] = vx
    dr[2] = vz
    dr[3] = γᵣ * (r̄ₘₚ - rₘₚ)
    dr[4] = γₕ * (z̄ₘₚ - zₘₚ)
end

function input_function!(id::PlanarGMAWDynamics, ds::AbstractVector{Ty}, r, u) where Ty
    N = length(s) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ, rₘₚ, zₘₚ = r[1], r[2], r[3], r[4]
    vx, vz, trim, WFS = u[1], u[2], u[3], u[4]
    TS = sqrt(vx^2 + vz^2)
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, T₀, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.T₀, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

    F = get!(id.F_cache, Ty) do
        zeros(Ty, n_rows*n_cols)
    end::Vector{Ty}

    # V = f(trim)
    # I = f(wfs, trim, v, ctwd)

    ṁ = WFS * π * (wire_diam / 2)^2 * ρ # kg/s
    P = 29000 * WFS # W/(m/s) * (m/s)

    @. F = normpdf((xₙ - xₜ) / l / wₓ) * √(max(rₘₚ^2 - (zₙ - zₘₚ)^2, 0)) * logistic((zₙ - zₘₚ) / l + bₕ)

    if sum(F) > 0
        F ./= sum(F) # Normalize for conservation purposes
    else
        F .= 0
    end

    # Forced / input dynamics
    @. dE += η * F * P              # Add in torch power
    @. dE += F * (cₚ * ṁ * T∞)      # Add in energy contribution from incoming wire (assume room temp)
    @. dm += F * ṁ
end