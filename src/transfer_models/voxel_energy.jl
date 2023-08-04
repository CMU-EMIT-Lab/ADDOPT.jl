using StatsFuns
using Symbolics

struct VoxelEnergyDynamics <: TransferDynamics
    nx::Int
    ny::Int
    nz::Int

    l::Float64

    xₙ::Vector{Float64}
    yₙ::Vector{Float64}
    zₙ::Vector{Float64}

    k::Float64
    ρ::Float64
    cₚ::Float64

    T∞::Float64
    T₀::Float64

    wire_diam::Float64

    h∞::Float64
    h₀::Float64
    Tmin::Float64
    Tmax::Float64

    x::Vector{Float64}

    C_cache::Dict{Tuple{DataType,Int},Any}
    B::Vector{Float64}

    function VoxelEnergyDynamics(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, x)
        C = Dict{Tuple{DataType,Int},Any}()

        B = zeros(nx, ny, nz)
        B[:, :, 1] .= 1
        B = reshape(B, nx * ny * nz)

        return new(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, x, C, B)
    end
end

@inline Ns(td::VoxelEnergyDynamics)::Int = td.nx * td.ny * td.nz
state_min(td::VoxelEnergyDynamics) = zeros(Ns(td))
state_max(td::VoxelEnergyDynamics) = Inf * ones(Ns(td))

function dynamics_function!(td::VoxelEnergyDynamics, ds::AbstractVector{Ty}, s, t) where {Ty}
    nz, ny, nx = td.nz, td.ny, td.nx
    l = td.l
    h∞, h₀ = td.h∞, td.h₀
    T∞, T₀ = td.T∞, td.T₀
    B = td.B
    k = td.k
    ρ, cₚ = td.ρ, td.cₚ
    x = td.x

    thread::Int = Threads.threadid()
    C = get!(td.C_cache, (Ty, thread)) do
        zeros(Ty, nz * ny * nx)
    end::Vector{Ty}

    N = Ns(td)
    dE = view(ds, 1:N)
    E = view(s, 1:N)

    rcE = reshape(E, (nx, ny, nz))
    rcdE = reshape(dE, (nx, ny, nz))
    rcx = reshape(x, (nx, ny, nz))
    rcC = reshape(view(C, :), (nx, ny, nz))

    dE .= 0.0
    C .= 6.0

    # From top
    dEᵢ = view(rcdE, :, :, 1:(nz-1))
    Eᵢ = view(rcE, :, :, 1:(nz-1))
    Eⱼ = view(rcE, :, :, 2:nz)
    xᵢ = view(rcx, :, :, 1:(nz-1))
    xⱼ = view(rcx, :, :, 2:nz)
    Cᵢ = view(rcC, :, :, 1:(nz-1))
    Cⱼ = view(rcC, :, :, 2:nz)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    # From bottom
    dEᵢ = view(rcdE, :, :, 2:nz)
    Eᵢ = view(rcE, :, :, 2:nz)
    Eⱼ = view(rcE, :, :, 1:(nz-1))
    xᵢ = view(rcx, :, :, 2:nz)
    xⱼ = view(rcx, :, :, 1:(nz-1))
    Cᵢ = view(rcC, :, :, 2:nz)
    Cⱼ = view(rcC, :, :, 1:(nz-1))
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    # From in
    dEᵢ = view(rcdE, :, 2:ny, :)
    Eᵢ = view(rcE, :, 2:ny, :)
    Eⱼ = view(rcE, :, 1:(ny-1), :)
    xᵢ = view(rcx, :, 2:ny, :)
    xⱼ = view(rcx, :, 1:(ny-1), :)
    Cᵢ = view(rcC, :, 2:ny, :)
    Cⱼ = view(rcC, :, 1:(ny-1), :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    # From out
    dEᵢ = view(rcdE, :, 1:(ny-1), :)
    Eᵢ = view(rcE, :, 1:(ny-1), :)
    Eⱼ = view(rcE, :, 2:ny, :)
    xᵢ = view(rcx, :, 1:(ny-1), :)
    xⱼ = view(rcx, :, 2:ny, :)
    Cᵢ = view(rcC, :, 1:(ny-1), :)
    Cⱼ = view(rcC, :, 2:ny, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    # From left
    dEᵢ = view(rcdE, 2:nx, :, :)
    Eᵢ = view(rcE, 2:nx, :, :)
    Eⱼ = view(rcE, 1:(nx-1), :, :)
    xᵢ = view(rcx, 2:nx, :, :)
    xⱼ = view(rcx, 1:(nx-1), :, :)
    Cᵢ = view(rcC, 2:nx, :, :)
    Cⱼ = view(rcC, 1:(nx-1), :, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    # From right
    dEᵢ = view(rcdE, 1:(nx-1), :, :)
    Eᵢ = view(rcE, 1:(nx-1), :, :)
    Eⱼ = view(rcE, 2:nx, :, :)
    xᵢ = view(rcx, 1:(nx-1), :, :)
    xⱼ = view(rcx, 2:nx, :, :)
    Cᵢ = view(rcC, 1:(nx-1), :, :)
    Cⱼ = view(rcC, 2:nx, :, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    Cᵢ .-= xⱼ

    dE .+= h∞ .* C .* (T∞ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Convection to environment

    ###################
    dE .+= h₀ .* B .* (T₀ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Conduction to baseplate

    # if ṁ > 0 && P > 0
    #     dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ .* T∞ .* m .- E)
    # end # Convection from argon
end

function temperature!(td::VoxelEnergyDynamics, T, s)
    N = Ns(td)
    E = view(s, 1:N)
    x = td.x

    map!((E, x) -> E / (td.l^3 * td.ρ * td.cₚ) + (1.0 - x) * 293.15, T, E, x)
end