using ExponentialUtilities

function discretize_linear_dynamics(A, B, e, dt)
    nx, nu = size(B)
    H = [A B I(nx);
        zeros(nu + nx, nx + nu + nx)] # Dynamics matrix for combined system of x and u
    H .*= dt
    G = exponential!(H) # State transition matrix for combined system

    Ad = G[1:nx, 1:nx]
    Bd = G[1:nx, (nx+1):(nx+nu)]
    ed = G[1:nx, (nx+nu+1):(2nx+nu)] * e

    return Ad, Bd, ed
end

function matrices_for_voxel_conduction(nx, ny, nz, l, α, C, h, T₀, T∞, σ, buffer)
    L = cu(voxel_laplacian(nx, ny, nz))
    A∞ = Diagonal(cu(surface_voxels(nx, ny, nz))) .* l^2
    A₀ = Diagonal(cu(volume_voxels(nx, ny, nz))) * l^2

    power_mask = zeros(Bool, (nx, ny, nz))
    power_mask[(1+buffer):(end-buffer), (1+buffer):(end-buffer), 1] .= true
    power_mask = cu(vec(power_mask))

    A = -((α / l^2) .* L .+ (α / l^4) .* A₀ .+ (h / C) .* A∞)
    B = (σ == 0.0) ? (A∞ ./ ((l^2) * C)) : (cu(pairwise_gaussian_integral(nx, ny, nz, l, σ)) .* cu(surface_voxels(nx, ny, nz)) ./ C)
    B = B[:, power_mask]
    e = ((α / l^4) .* A₀ .* T₀ + (h / C) .* A∞ .* T∞) * cu(ones(nx * ny * nz))

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