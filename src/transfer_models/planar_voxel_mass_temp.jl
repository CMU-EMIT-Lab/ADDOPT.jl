using StatsFuns
using Symbolics

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

    C_cache::Dict{Tuple{DataType, Int},Any}
    K_cache::Dict{Tuple{DataType, Int},Any}

    # A::Vector{Float64}
    B::Vector{Float64}

    function PlanarVoxelMassEnergyDynamics(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
        C = Dict{Tuple{DataType, Int},Any}()
        K = Dict{Tuple{DataType, Int},Any}()

        # A = 2l^2 * ones(nrows * ncols)
        B = vcat(ones(ncols), zeros((nrows - 1) * ncols))

        return new(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, C, K, B)
    end
end

@inline Ns(td::PlanarVoxelMassEnergyDynamics)::Int = td.nrows * td.ncols * 2
state_min(td::PlanarVoxelMassEnergyDynamics) = zeros(Ns(td))
state_max(td::PlanarVoxelMassEnergyDynamics) = Inf * ones(Ns(td))

function dynamics_function!(td::PlanarVoxelMassEnergyDynamics, ds::AbstractVector{Ty}, s, t, zi) where {Ty}
    nrows, ncols = td.nrows, td.ncols
    l, w = td.l, td.w
    h∞, h₀ = td.h∞, td.h₀
    T∞, T₀ = td.T∞, td.T₀
    B = td.B
    k = td.kₘ
    ρ, cₚ = td.ρₘ, td.cₚₘ

    thread::Int = Threads.threadid()
    C = get!(td.C_cache, (Ty, thread)) do
        zeros(Ty, nrows * ncols)
    end::Vector{Ty}
    K = get!(td.K_cache, (Ty, thread)) do
        zeros(Ty, nrows * ncols)
    end::Vector{Ty}

    N = Ns(td) ÷ 2
    dE = view(ds, 1:N)
    dx = view(ds, (N+1):2N)
    E = view(s, 1:N)
    x = view(s, (N+1):2N)

    # map!((x) -> conductivity(td, x), K, x)
    # map!((x) -> heat_capacity(td, x), C, x)

    μ(x) = typeof(x) == Symbolics.Num ? (x - 1)^2 : (x > 1 ? (x - 1)^2 : 0)

    rcE = reshape(E, (ncols, nrows))
    rcdE = reshape(dE, (ncols, nrows))
    rcdx = reshape(dx, (ncols, nrows))
    rcx = reshape(x, (ncols, nrows))
    rcC = reshape(view(C, :), (ncols, nrows))
    rcK = reshape(view(K, :), (ncols, nrows))

    dx .= 0.0
    dE .= 0.0
    C .= 6

    # From top
    dEᵢ = view(rcdE, :, 1:(nrows-1))
    dxᵢ = view(rcdx, :, 1:(nrows-1))
    Eᵢ = view(rcE, :, 1:(nrows-1))
    Eⱼ = view(rcE, :, 2:nrows)
    xᵢ = view(rcx, :, 1:(nrows-1))
    xⱼ = view(rcx, :, 2:nrows)
    Cᵢ = view(rcC, :, 1:(nrows-1))
    Cⱼ = view(rcC, :, 2:nrows)
    Kᵢ = view(rcK, :, 1:(nrows-1))
    Kⱼ = view(rcK, :, 2:nrows)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) - μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From bottom
    dEᵢ = view(rcdE, :, 2:nrows)
    dxᵢ = view(rcdx, :, 2:nrows)
    Eᵢ = view(rcE, :, 2:nrows)
    Eⱼ = view(rcE, :, 1:(nrows-1))
    xᵢ = view(rcx, :, 2:nrows)
    xⱼ = view(rcx, :, 1:(nrows-1))
    Cᵢ = view(rcC, :, 2:nrows)
    Cⱼ = view(rcC, :, 1:(nrows-1))
    Kᵢ = view(rcK, :, 2:nrows)
    Kⱼ = view(rcK, :, 1:(nrows-1))
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) - μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From left
    dEᵢ = view(rcdE, 2:ncols, :)
    dxᵢ = view(rcdx, 2:ncols, :)
    Eᵢ = view(rcE, 2:ncols, :)
    Eⱼ = view(rcE, 1:(ncols-1), :)
    xᵢ = view(rcx, 2:ncols, :)
    xⱼ = view(rcx, 1:(ncols-1), :)
    Cᵢ = view(rcC, 2:ncols, :)
    Cⱼ = view(rcC, 1:(ncols-1), :)
    Kᵢ = view(rcK, 2:ncols, :)
    Kⱼ = view(rcK, 1:(ncols-1), :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) - μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From right
    dEᵢ = view(rcdE, 1:(ncols-1), :)
    dxᵢ = view(rcdx, 1:(ncols-1), :)
    Eᵢ = view(rcE, 1:(ncols-1), :)
    Eⱼ = view(rcE, 2:ncols, :)
    xᵢ = view(rcx, 1:(ncols-1), :)
    xⱼ = view(rcx, 2:ncols, :)
    Cᵢ = view(rcC, 1:(ncols-1), :)
    Cⱼ = view(rcC, 2:ncols, :)
    Kᵢ = view(rcK, 1:(ncols-1), :)
    Kⱼ = view(rcK, 2:ncols, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) - μ.(xᵢ)
    Cᵢ .-= xⱼ

    dE .+= h∞ .* C .* (T∞ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Convection to environment

    ###################
    dE .+= h₀ .* B .* (T∞ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Conduction to baseplate

    # if ṁ > 0 && P > 0
    #     dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ .* T∞ .* m .- E)
    # end # Convection from argon
end

function temperature!(td::PlanarVoxelMassEnergyDynamics, T, s, zi)
    N = Ns(td) ÷ 2
    E = view(s, 1:N)
    x = view(s, (N+1):2N)

    T .= E ./ (td.l^3 * td.ρₘ * td.cₚₘ * x)
end

# conductivity(td::PlanarVoxelMassEnergyDynamics, m) = (td.kₘ - td.kₐ) * (m / (td.l^2 * td.w) - td.ρₐ) / (td.ρₘ - td.ρₐ) + td.kₐ
# heat_capacity(td::PlanarVoxelMassEnergyDynamics, m) = ((td.ρₘ * td.cₚₘ - td.ρₐ * td.cₚₐ) * (m / (td.l^2 * td.w) - td.ρₐ) / (td.ρₘ - td.ρₐ) + td.ρₐ * td.cₚₐ) * td.l^2 * td.w