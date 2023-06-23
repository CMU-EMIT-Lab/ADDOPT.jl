using StatsFuns

struct PlanarVoxelMassEnergyDynamics <: TransferDynamics
    nrows::Int
    ncols::Int

    l::Float64
    w::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    kₘ::Float64
    kₐ::Float64

    ρₘ::Float64
    ρₐ::Float64

    cₚₘ::Float64
    cₚₐ::Float64

    T∞::Float64
    T₀::Float64

    wire_diam::Float64

    h∞::Float64
    h₀::Float64
    Tmin::Float64
    Tmax::Float64

    C_cache::Dict{DataType,Any}
    K_cache::Dict{DataType,Any}

    A::Vector{Float64}
    B::Vector{Float64}

    function PlanarVoxelMassEnergyDynamics(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
        C = Dict{DataType,Any}()
        K = Dict{DataType,Any}()

        A = 2l^2 * ones(nrows * ncols)
        B = vcat(l * w * ones(ncols), zeros((nrows - 1) * ncols))

        return new(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, C, K, A, B)
    end
end

@inline Ns(td::PlanarVoxelMassEnergyDynamics)::Int = td.nrows * td.ncols * 2
state_min(td::PlanarVoxelMassEnergyDynamics) = [td.l^2 * td.w * td.ρₐ * td.cₚₐ * td.Tmin * ones(Ns(td) ÷ 2); td.l^2 * td.w * td.ρₐ * ones(Ns(td) ÷ 2)]
state_max(td::PlanarVoxelMassEnergyDynamics) = [td.l^2 * td.w * td.ρₘ * td.cₚₘ * td.Tmax * ones(Ns(td) ÷ 2); td.l^2 * td.w * td.ρₘ * ones(Ns(td) ÷ 2)]

function dynamics_function!(td::PlanarVoxelMassEnergyDynamics, ds::AbstractVector{Ty}, s, t) where {Ty}
    nrows, ncols = td.nrows, td.ncols
    l, w = td.l, td.w
    h∞, h₀ = td.h∞, td.h₀
    T∞, T₀ = td.T∞, td.T₀
    A, B = td.A, td.B

    C = get!(td.C_cache, Ty) do
        zeros(Ty, nrows * ncols)
    end::Vector{Ty}
    K = get!(td.K_cache, Ty) do
        zeros(Ty, nrows * ncols)
    end::Vector{Ty}

    N = Ns(td) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    E = view(s, 1:N)
    m = view(s, (N+1):2N)

    map!((x) -> conductivity(td, x), K, m)
    map!((x) -> heat_capacity(td, x), C, m)

    rcE = reshape(E, (ncols, nrows))
    rcdE = reshape(dE, (ncols, nrows))
    rcC = reshape(view(C, :), (ncols, nrows))
    rcK = reshape(view(K, :), (ncols, nrows))

    dm .= 0.
    dE .= 0.

    # From top
    dEᵢ = view(rcdE, :, 1:(nrows-1))
    Eᵢ = view(rcE, :, 1:(nrows-1))
    Eⱼ = view(rcE, :, 2:nrows)
    Cᵢ = view(rcC, :, 1:(nrows-1))
    Cⱼ = view(rcC, :, 2:nrows)
    Kᵢ = view(rcK, :, 1:(nrows-1))
    Kⱼ = view(rcK, :, 2:nrows)
    dEᵢ .+= 2w .* (Kᵢ .* Kⱼ) ./ (Kᵢ .+ Kⱼ) .* (Eⱼ ./ Cⱼ .- Eᵢ ./ Cᵢ)

    # From bottom
    dEᵢ = view(rcdE, :, 2:nrows)
    Eᵢ = view(rcE, :, 2:nrows)
    Eⱼ = view(rcE, :, 1:(nrows-1))
    Cᵢ = view(rcC, :, 2:nrows)
    Cⱼ = view(rcC, :, 1:(nrows-1))
    Kᵢ = view(rcK, :, 2:nrows)
    Kⱼ = view(rcK, :, 1:(nrows-1))
    dEᵢ .+= 2w .* (Kᵢ .* Kⱼ) ./ (Kᵢ .+ Kⱼ) .* (Eⱼ ./ Cⱼ .- Eᵢ ./ Cᵢ)

    # From left
    dEᵢ = view(rcdE, 2:ncols, :)
    Eᵢ = view(rcE, 2:ncols, :)
    Eⱼ = view(rcE, 1:(ncols-1), :)
    Cᵢ = view(rcC, 2:ncols, :)
    Cⱼ = view(rcC, 1:(ncols-1), :)
    Kᵢ = view(rcK, 2:ncols, :)
    Kⱼ = view(rcK, 1:(ncols-1), :)
    dEᵢ .+= 2w .* (Kᵢ .* Kⱼ) ./ (Kᵢ .+ Kⱼ) .* (Eⱼ ./ Cⱼ .- Eᵢ ./ Cᵢ)

    # From right
    dEᵢ = view(rcdE, 1:(ncols-1), :)
    Eᵢ = view(rcE, 1:(ncols-1), :)
    Eⱼ = view(rcE, 2:ncols, :)
    Cᵢ = view(rcC, 1:(ncols-1), :)
    Cⱼ = view(rcC, 2:ncols, :)
    Kᵢ = view(rcK, 1:(ncols-1), :)
    Kⱼ = view(rcK, 2:ncols, :)
    dEᵢ .+= 2w .* (Kᵢ .* Kⱼ) ./ (Kᵢ .+ Kⱼ) .* (Eⱼ ./ Cⱼ .- Eᵢ ./ Cᵢ)

    dE .+= h∞ .* A .* (T∞ .- E ./ C) # Convection to environment
    dE .+= h₀ .* B .* (T₀ .- E ./ C) # Conduction to baseplate

    # if ṁ > 0 && P > 0
    #     dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ .* T∞ .* m .- E)
    # end # Convection from argon
end

function temperature!(td::PlanarVoxelMassEnergyDynamics, T, s)
    N = Ns(td) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    hc(x) = heat_capacity(td, x)

    T .= E ./ hc.(m)
end

conductivity(td::PlanarVoxelMassEnergyDynamics, m) = (td.kₘ - td.kₐ) * (m / (td.l^2 * td.w) - td.ρₐ) / (td.ρₘ - td.ρₐ) + td.kₐ
heat_capacity(td::PlanarVoxelMassEnergyDynamics, m) = ((td.ρₘ * td.cₚₘ - td.ρₐ * td.cₚₐ) * (m / (td.l^2 * td.w) - td.ρₐ) / (td.ρₘ - td.ρₐ) + td.ρₐ * td.cₚₐ) * td.l^2 * td.w