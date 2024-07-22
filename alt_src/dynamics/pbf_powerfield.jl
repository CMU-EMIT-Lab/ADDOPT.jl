struct PBFPowerField{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Dynamics{T}
    nx::Int
    ny::Int
    nz::Int

    l::T

    k::T
    ρ::T
    cₚ::T

    T∞::T
    T₀::T

    h::T

    A::M
    B::M
    e::V

    Ad::M
    Bd::M
    ed::V

    Δt::T

    σ::T
    buffer::Int

    function PBFPowerField(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, h, Δt, T, V, M; σ=0.0, buffer=0)
        α = k / ρ / cₚ
        C = ρ * l^3 * cₚ

        A, B, e = matrices_for_voxel_conduction(nx, ny, nz, l, α, C, h, T₀, T∞, σ, buffer)
        Ad, Bd, ed = discretize_linear_dynamics(A, B, e, Δt)

        new{T,V,M}(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, h, A, B, e, Ad, Bd, ed, Δt, σ, buffer)
    end
end

nx(dynamics::PBFPowerField)::Int = dynamics.nx * dynamics.ny * dynamics.nz
nu(dynamics::PBFPowerField)::Int = (dynamics.nx - 2dynamics.buffer) * (dynamics.ny - 2dynamics.buffer)
Δt(dynamics::PBFPowerField, u) = dynamics.Δt

function rate!(dynamics::PBFPowerField{T}, ẋ::AbstractVector{E}, x, u) where {T,E}
    A, B = dynamics.A, dynamics.B

    mul!(ẋ, A, x, 1.0, 1.0)
    mul!(ẋ, B, u, 1.0, 1.0)
end

function transition!(dynamics::PBFPowerField, xₖ₊₁::AbstractVector{E}, xₖ, uₖ) where {E}
    Ad, Bd, ed = dynamics.Ad, dynamics.Bd, dynamics.ed

    xₖ₊₁ .= ed
    mul!(xₖ₊₁, Ad, xₖ, 1.0, 1.0)
    mul!(xₖ₊₁, Bd, uₖ, 1.0, 1.0)
end

function transition_state_jacobian!(dynamics::PBFPowerField, A, xₖ, uₖ)
    A .= dynamics.Ad
end

function transition_input_jacobian!(dynamics::PBFPowerField, B, xₖ, uₖ)
    B .= dynamics.Bd
end

function discretize_linear_dynamics(A, B, e, dt)
    nx, nu = size(B)
    H = [A B I(nx);
        zeros(nu + nx, nx + nu + nx)] # Dynamics matrix for combined system of x and u
    G = exp(H * dt) # State transition matrix for combined system

    Ad = G[1:nx, 1:nx]
    Bd = G[1:nx, (nx+1):(nx+nu)]
    ed = G[1:nx, (nx+nu+1):(2nx+nu)] * e

    return Ad, Bd, ed
end

function matrices_for_voxel_conduction(nx, ny, nz, l, α, C, h, T₀, T∞, σ, buffer)
    L = voxel_laplacian(nx, ny, nz)
    A∞ = Diagonal(surface_voxels(nx, ny, nz)) * l^2
    A₀ = Diagonal(volume_voxels(nx, ny, nz)) * l^2

    power_mask = zeros(Bool, (nx, ny, nz))
    power_mask[(1+buffer):(end-buffer), (1+buffer):(end-buffer), 1] .= true
    power_mask = vec(power_mask)

    A = -((α / l^2) * L + (α / l^4) * A₀ + (h / C) * A∞)
    B = (σ == 0.0) ? (A∞ / (l^2) / C) : (pairwise_gaussian_integral(nx, ny, nz, l, σ) .* surface_voxels(nx, ny, nz) ./ C)
    B = B[:, power_mask]
    e = ((α / l^4) * A₀ * T₀ + (h / C) * A∞ * T∞) * ones(nx * ny * nz)

    return A, B, e
end

function surface_voxels(nx, ny, nz)
    voxels = zeros(nx, ny, nz)
    voxels[:, :, 1] .= 1.0

    return vec(voxels)
end

function volume_voxels(nx, ny, nz)
    voxels = zeros(nx, ny, nz)
    voxels[:, :, end] .+= 1.0
    voxels[1, :, :] .+= 1.0
    voxels[end, :, :] .+= 1.0
    voxels[:, 1, :] .+= 1.0
    voxels[:, end, :] .+= 1.0

    return vec(voxels)
end

function voxel_laplacian(nx, ny, nz)
    A = voxel_adjacency(nx, ny, nz)
    D = voxel_degree(A)
    return D - A
end

function voxel_degree(A)
    d = sum(A, dims=2)[:, 1]
    return Diagonal(d)
end

function voxel_adjacency(nx, ny, nz)
    A = zeros(nx * ny * nz, nx * ny * nz)
    idx = LinearIndices(zeros(nx, ny, nz))

    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                if i < nx
                    A[idx[i, j, k], idx[i+1, j, k]] = 1
                end
                if i > 1
                    A[idx[i, j, k], idx[i-1, j, k]] = 1
                end
                if j < ny
                    A[idx[i, j, k], idx[i, j+1, k]] = 1
                end
                if j > 1
                    A[idx[i, j, k], idx[i, j-1, k]] = 1
                end
                if k < nz
                    A[idx[i, j, k], idx[i, j, k+1]] = 1
                end
                if k > 1
                    A[idx[i, j, k], idx[i, j, k-1]] = 1
                end
            end
        end
    end

    return A
end

function get_coordinates(nx, ny, nz, l)
    x = zeros(nx, ny, nz)
    y = zeros(nx, ny, nz)
    z = zeros(nx, ny, nz)

    for i in 1:nx
        x[i, :, :] .= i * l
    end
    for i in 1:ny
        y[:, i, :] .= i * l
    end
    for i in 1:nz
        z[:, :, i] .= i * l
    end

    return [vec(x)'; vec(y)'; vec(z)']
end

function pairwise_distances(nx, ny, nz, l)
    xyz = get_coordinates(nx, ny, nz, l)
    nvox = nx * ny * nz
    D = zeros(nvox, nvox)
    dr = zeros(3)

    for i in 1:nvox
        for j in 1:nvox
            dr .= xyz[:, i] .- xyz[:, j]
            D[i, j] = norm(dr)
        end
    end

    return D
end

function pairwise_gaussian_integral(nx, ny, nz, l, σ)
    D = pairwise_distances(nx, ny, nz, l)
    G = map(d -> (l^2 / (2π * σ^2)) * exp(-(d^2 / (2 * σ^2))), D)

    return G
end