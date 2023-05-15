struct PlanarGMAWDynamics <: InputDynamics
    l
    xₙ
    zₙ

    k
    ρ
    cₚ
    T∞
    wire_diam

    h∞
    h₀
    hₐᵣ
    η
    γᵣ
    γₕ
    wₓ
    bₕ

    F
    ZW

    function PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
        F = zeros(nrows * ncols)
        ZW = zeros(nrows * ncols)

        return new(l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, F, ZW)
    end
end

Nu(id::PlanarGMAWDynamics) = 5 # torch position (x,z), trim, WFS, TS
Nr(id::PlanarGMAWDynamics) = 2 # meltpool root and radius

input_min(id::PlanarGMAWDynamics) = [-Inf; -Inf; 0.5; 0; 0]
input_max(id::PlanarGMAWDynamics) = [Inf; Inf; 1.2; 120.0; 20.0]

state_min(id::PlanarGMAWDynamics) = [0.0; 0.0]
state_max(id::PlanarGMAWDynamics) = [Inf; Inf]

function dynamics_function!(id::PlanarGMAWDynamics, dr, s, r, u)
    N = length(s) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    rₘₚ, zₘₚ = r[1], r[2]
    xₜ, zₜ, trim, WFS, TS = u[1], u[2], u[3], u[4], u[5]
    ZW, l, xₙ, zₙ = id.ZW, id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, T₀, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.T₀, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

    z̄ₘₚ = zₘₚ
    # Compute steady state meltpool radius and z location
    r̄ₘₚ = WFS > 0 && TS > 0 ? wire_diam * √(WFS / TS) * √(1 / 2) : wire_diam
    @. ZW = normpdf((xₙ - xₜ) / l * 2) * (1 - exp(-m / (ρ * l^2))) * max(sign(Tₗ * m * cₚ - E), 0) * exp((zₙ / l) * 10)
    ZW_sum = sum(ZW) # logistic(1000*(Tₗ*m*cₚ - E) - 20)
    if ZW_sum > 0 && WFS > 0 && TS > 0
        ZW .*= zₙ
        z̄ₘₚ = sum(ZW) / ZW_sum + l
    end

    dr[1] = γᵣ * (r̄ₘₚ - rₘₚ)
    dr[2] = γₕ * (z̄ₘₚ - zₘₚ)
end

function input_function!(id::PlanarGMAWDynamics, ds, r, u)
    N = length(s) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    rₘₚ, zₘₚ = r[1], r[2]
    xₜ, zₜ, trim, WFS, TS = u[1], u[2], u[3], u[4], u[5]
    F, l, xₙ, zₙ = id.F, id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, T₀, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.T₀, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

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