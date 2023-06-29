using StatsFuns
using Symbolics

struct VoxelEnergyFillDynamics <: TransferDynamics
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

    C_cache::Dict{DataType,Any}
    K_cache::Dict{DataType,Any}

    B::Vector{Float64}

    function VoxelEnergyFillDynamics(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
        C = Dict{DataType,Any}()
        K = Dict{DataType,Any}()

        B = zeros(nx, ny, nz)
        B[:, :, 1] .= 1
        B = reshape(B, nx * ny * nz)

        return new(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, C, K, B)
    end
end

@inline Ns(td::VoxelEnergyFillDynamics)::Int = td.nx * td.ny * td.nz * 2
state_min(td::VoxelEnergyFillDynamics) = zeros(Ns(td))
state_max(td::VoxelEnergyFillDynamics) = Inf * ones(Ns(td))

function dynamics_function!(td::VoxelEnergyFillDynamics, ds::AbstractVector{Ty}, s, t) where {Ty}
    nz, ny, nx = td.nz, td.ny, td.nx
    l = td.l
    h∞, h₀ = td.h∞, td.h₀
    T∞, T₀ = td.T∞, td.T₀
    B = td.B
    k = td.k
    ρ, cₚ = td.ρ, td.cₚ

    C = get!(td.C_cache, Ty) do
        zeros(Ty, nz * ny * nx)
    end::Vector{Ty}
    K = get!(td.K_cache, Ty) do
        zeros(Ty, nz * ny * nx)
    end::Vector{Ty}

    N = Ns(td) ÷ 2
    dE = view(ds, 1:N)
    dx = view(ds, (N+1):2N)
    E = view(s, 1:N)
    x = view(s, (N+1):2N)

    μ(x) = typeof(x) == Symbolics.Num ? 10 * (x - 1)^2 : (x > 1 ? 10 * (x - 1)^2 : 0)

    rcE = reshape(E, (nx, ny, nz))
    rcdE = reshape(dE, (nx, ny, nz))
    rcdx = reshape(dx, (nx, ny, nz))
    rcx = reshape(x, (nx, ny, nz))
    rcC = reshape(view(C, :), (nx, ny, nz))
    rcK = reshape(view(K, :), (nx, ny, nz))

    dx .= 0.0
    dE .= 0.0
    C .= 6.0

    # From top
    dEᵢ = view(rcdE, :, :, 1:(nz-1))
    dxᵢ = view(rcdx, :, :, 1:(nz-1))
    Eᵢ = view(rcE, :, :, 1:(nz-1))
    Eⱼ = view(rcE, :, :, 2:nz)
    xᵢ = view(rcx, :, :, 1:(nz-1))
    xⱼ = view(rcx, :, :, 2:nz)
    Cᵢ = view(rcC, :, :, 1:(nz-1))
    Cⱼ = view(rcC, :, :, 2:nz)
    Kᵢ = view(rcK, :, :, 1:(nz-1))
    Kⱼ = view(rcK, :, :, 2:nz)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From bottom
    dEᵢ = view(rcdE, :, :, 2:nz)
    dxᵢ = view(rcdx, :, :, 2:nz)
    Eᵢ = view(rcE, :, :, 2:nz)
    Eⱼ = view(rcE, :, :, 1:(nz-1))
    xᵢ = view(rcx, :, :, 2:nz)
    xⱼ = view(rcx, :, :, 1:(nz-1))
    Cᵢ = view(rcC, :, :, 2:nz)
    Cⱼ = view(rcC, :, :, 1:(nz-1))
    Kᵢ = view(rcK, :, :, 2:nz)
    Kⱼ = view(rcK, :, :, 1:(nz-1))
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From in
    dEᵢ = view(rcdE, :, 2:ny, :)
    dxᵢ = view(rcdx, :, 2:ny, :)
    Eᵢ = view(rcE, :, 2:ny, :)
    Eⱼ = view(rcE, :, 1:(ny-1), :)
    xᵢ = view(rcx, :, 2:ny, :)
    xⱼ = view(rcx, :, 1:(ny-1), :)
    Cᵢ = view(rcC, :, 2:ny, :)
    Cⱼ = view(rcC, :, 1:(ny-1), :)
    Kᵢ = view(rcK, :, 2:ny, :)
    Kⱼ = view(rcK, :, 1:(ny-1), :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From out
    dEᵢ = view(rcdE, :, 1:(ny-1), :)
    dxᵢ = view(rcdx, :, 1:(ny-1), :)
    Eᵢ = view(rcE, :, 1:(ny-1), :)
    Eⱼ = view(rcE, :, 2:ny, :)
    xᵢ = view(rcx, :, 1:(ny-1), :)
    xⱼ = view(rcx, :, 2:ny, :)
    Cᵢ = view(rcC, :, 1:(ny-1), :)
    Cⱼ = view(rcC, :, 2:ny, :)
    Kᵢ = view(rcK, :, 1:(ny-1), :)
    Kⱼ = view(rcK, :, 2:ny, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From left
    dEᵢ = view(rcdE, 2:nx, :, :)
    dxᵢ = view(rcdx, 2:nx, :, :)
    Eᵢ = view(rcE, 2:nx, :, :)
    Eⱼ = view(rcE, 1:(nx-1), :, :)
    xᵢ = view(rcx, 2:nx, :, :)
    xⱼ = view(rcx, 1:(nx-1), :, :)
    Cᵢ = view(rcC, 2:nx, :, :)
    Cⱼ = view(rcC, 1:(nx-1), :, :)
    Kᵢ = view(rcK, 2:nx, :, :)
    Kⱼ = view(rcK, 1:(nx-1), :, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    # From right
    dEᵢ = view(rcdE, 1:(nx-1), :, :)
    dxᵢ = view(rcdx, 1:(nx-1), :, :)
    Eᵢ = view(rcE, 1:(nx-1), :, :)
    Eⱼ = view(rcE, 2:nx, :, :)
    xᵢ = view(rcx, 1:(nx-1), :, :)
    xⱼ = view(rcx, 2:nx, :, :)
    Cᵢ = view(rcC, 1:(nx-1), :, :)
    Cⱼ = view(rcC, 2:nx, :, :)
    Kᵢ = view(rcK, 1:(nx-1), :, :)
    Kⱼ = view(rcK, 2:nx, :, :)
    dEᵢ .+= (k / (l^2 * ρ * cₚ)) .* (Eⱼ .* xᵢ .- Eᵢ .* xⱼ)
    dxᵢ .+= μ.(xⱼ) .- μ.(xᵢ)
    Cᵢ .-= xⱼ

    dE .+= h∞ .* C .* (T∞ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Convection to environment

    ###################
    dE .+= h₀ .* B .* (T₀ .* x .* l^2 .- E ./ (ρ * l * cₚ)) # Conduction to baseplate

    # if ṁ > 0 && P > 0
    #     dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ .* T∞ .* m .- E)
    # end # Convection from argon
end

function temperature!(td::VoxelEnergyFillDynamics, T, s)
    N = Ns(td) ÷ 2
    E = view(s, 1:N)
    x = view(s, (N+1):2N)

    T .= E ./ (td.l^3 * td.ρ * td.cₚ * x)
end