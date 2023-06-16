using StatsFuns

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

    F_cache::Dict{DataType,Any}
    ZW_cache::Dict{DataType,Any}

    function PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
        F = Dict{DataType,Any}()
        ZW = Dict{DataType,Any}()

        return new(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, F, ZW)
    end
end

Nu(id::PlanarGMAWDynamics)::Int = 4 # vx, vz, trim, WFS (m/s for speeds)
Nr(id::PlanarGMAWDynamics)::Int = 4 # torch position (x,z), meltpool radius, meltpool root (z) (m)

input_min(id::PlanarGMAWDynamics) = [-0.01; 0.0; 0.5; 0.001] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::PlanarGMAWDynamics) = [0.01; (id.nrows+1)*id.l; 1.2; 0.1]
input_idle(id::PlanarGMAWDynamics) = [0.0; 0.0; 0.0; 0.0]

state_min(id::PlanarGMAWDynamics) = [0.0; 0.0; id.l; 0.0]
state_max(id::PlanarGMAWDynamics) = [(id.ncols+1)*id.l; (id.nrows+1)*id.l; id.l*10; (id.nrows+1)*id.l]

function dynamics_function!(id::PlanarGMAWDynamics, dr::AbstractVector{Ty}, s, r, u, t) where {Ty}
    N = length(s) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ, rₘₚ, zₘₚ = r[1], r[2], r[3], r[4]
    vx, vz, trim, WFS = u[1], u[2], u[3], u[4]
    TS = sqrt(vx^2)# + vz^2)
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, Tₗ, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.Tₗ, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

    ZW = get!(id.ZW_cache, Ty) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    z̄ₘₚ = zₜ#zₘₚ
    # pow_n = 3
    # Compute steady state meltpool radius and z location
    r̄ₘₚ = wire_diam * √((WFS + .001) / (TS + .001)) * √(1 / 2) #: wire_diam # (eltype(ZW) == Symbolics.Num) || (WFS > 0 && TS > 0) ? 
    # @. ZW = normpdf((xₙ - xₜ) / l * 2) * (1 - exp(-m / (ρ * l^2) * 2000)) * logistic(1000 * (Tₗ * m * cₚ - E)) * (zₙ / l)^pow_n
    # ZW_sum = sum(ZW) + 0.1#   #* max(sign(Tₗ * m * cₚ - E), 0) * exp((zₙ / l) * 10)
    # # if typeof(ZW_sum) == Symbolics.Num
    # #     z̄ₘₚ = sum(ZW)
    # # elseif ZW_sum > 0 && WFS > 0 && TS > 0
    #     ZW .*= zₙ
    #     z̄ₘₚ = ((sum(ZW)+ 0.1*l) / ZW_sum)*((pow_n + 2)/(pow_n + 1)) #+ l
    # # end

    # @show ZW_sum

    dr[1] = vx
    dr[2] = 8 * (vz - zₜ) 
    dr[3] = γᵣ * (r̄ₘₚ - rₘₚ)
    dr[4] = γₕ * (z̄ₘₚ - zₘₚ)
end

function input_function!(id::PlanarGMAWDynamics, ds::AbstractVector{Ty}, r, u, t) where {Ty}
    N = length(ds) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ, rₘₚ, zₘₚ = r[1], r[2], r[3], r[4]
    vx, vz, trim, WFS = u[1], u[2], u[3], u[4]
    TS = sqrt(vx^2 + vz^2)
    l, xₙ, zₙ = id.l, id.xₙ, id.zₙ
    k, ρ, cₚ, T∞, wire_diam = id.k, id.ρ, id.cₚ, id.T∞, id.wire_diam
    h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ = id.h∞, id.h₀, id.hₐᵣ, id.η, id.γᵣ, id.γₕ, id.wₓ, id.bₕ

    F = get!(id.F_cache, Ty) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    # V = f(trim)
    # I = f(wfs, trim, v, ctwd)

    ṁ = WFS * π * (wire_diam / 2)^2 * ρ # kg/s
    P = 29000 * WFS # W/(m/s) * (m/s)

    @. F = (l^2 / (2π*(l/wₓ)*rₘₚ)) * exp(-((xₙ - xₜ)^2/(l/wₓ)^2 + (zₙ - zₘₚ)^2/rₘₚ^2)/2)
    # @. F = normpdf((xₙ - xₜ) / l * wₓ) * normpdf((zₙ - zₘₚ) / rₘₚ) #* logistic((zₙ - zₘₚ) / l + bₕ) 
    # @show sum(F)

    # if eltype(F) == Symbolics.Num
    #     @. F += xₜ + zₜ + rₘₚ + zₘₚ + WFS
    # else
    #     F ./= sum(F)
    # end 

    # else
    #     @. F *= √(max(rₘₚ^2 - (zₙ - zₘₚ)^2, 0))
    # end

    # if eltype(F) == Symbolics.Num
    #     F .= 1
    # else
    # elseif sum(F) > 0
        # F ./= sum(F) # Normalize for conservation purposes ###################
    # else
        # F .= 0
    # end

    # @show F

    # Forced / input dynamics
    @. dE += η * F * P              # Add in torch power
    @. dE += F * (cₚ * ṁ * T∞)      # Add in energy contribution from incoming wire (assume room temp)
    @. dm += F * ṁ
end