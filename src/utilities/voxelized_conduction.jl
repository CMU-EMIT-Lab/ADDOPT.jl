function discretize_linear_dynamics(A::M, B, e, dt) where {M}
    nx, nu = size(B)
    H = vcat([A M(B) M(I(nx))],
        M(zeros(nu + nx, nx + nu + nx))) # Dynamics matrix for combined system of x and u
    H .*= dt
    G = exponential!(H) # State transition matrix for combined system

    Ad = G[1:nx, 1:nx]
    Bd = G[1:nx, (nx+1):(nx+nu)]
    ed = G[1:nx, (nx+nu+1):(2nx+nu)] * e

    return Ad, Bd, ed
end

function matrices_for_voxel_conduction(nx, ny, nz, l, α, C, h, T₀, T∞, σ, buffer)
    L = voxel_laplacian(nx, ny, nz)
    A∞ = Diagonal(surface_voxels(nx, ny, nz)) .* l^2
    A₀ = Diagonal(volume_voxels(nx, ny, nz)) .* l^2

    power_mask = zeros(Bool, (nx, ny, nz))
    power_mask[(1+buffer):(end-buffer), (1+buffer):(end-buffer), 1] .= true
    power_mask = vec(power_mask)

    A = -((α / l^2) .* L .+ (α / l^4) .* A₀ .+ (h / C) .* A∞)
    B = (σ == 0.0) ? (A∞ ./ ((l^2) * C)) : (pairwise_gaussian_integral(nx, ny, nz, l, σ) .* surface_voxels(nx, ny, nz) ./ C)
    B = B[:, power_mask]
    e = ((α / l^4) .* A₀ .* T₀ .+ (h / C) .* A∞ .* T∞) * ones(nx * ny * nz)

    return cu(A), cu(B), cu(e)
end

# For use with axisymmetric cylindrical coordinate setups
function matrices_for_voxel_conduction(nz, nr, Δr, Δz, k, ρ, cₚ, h, T∞, buffer)
    R, Z = cu(voxel_laplacian_directional(nz, nr)) # For second derivative approximation directional
    R1 = cu(horizontal_adjacency(nz, nr)) # For first derivative approximation radial direction
    Rp, _, Zp, Zn = cu(all_2D_surface_voxels(nz, nr)) # Surfaces in order: Right, Left, Top, Bottom (noted by positive (p) and negative (n))

    # Making an array of r values
    r = r_vector(nz, nr, Δr)

    α = cu(k ./ ρ ./ cₚ)

    # Mask for where power is applied
    power_mask = zeros(Bool, (nz, nr))
    # power_mask[1, 1:(end-buffer)] .= true # Inner edge doesn't need a buffer
    power_mask[1, (1+buffer[1]):(end-buffer[2])] .= true
    power_mask = cu(vec(power_mask))

    h2 = 0.0 # kW/m2kK # Heat transfer to outer surface

    e = cu(zeros(nz*nr))
    e .+= h ./ (ρ .* cₚ*Δz) .* Zp * T∞ # Convection top surface
    e .+= (h2*2*nr*Δr) ./ (ρ .* cₚ*(2*nr*Δr*Δr .- Δr*Δr)) .* Rp .* T∞ # Heat transfer to outer surface

    Rp = Diagonal(Rp)
    Zp = Diagonal(Zp)
    Zn = Diagonal(Zn)

    idx = LinearIndices((nz, nr))
    A = zeros(nz*nr, nz*nr)
    Az = π * (2 * r * Δr .- Δr^2) # Area in Z direction 
    Arp = 2 * π * r * Δz # Area in R direction outwards
    Arn = 2 * π * (r .- Δr) * Δz # Area in R direction inwards
    C = Array(ρ) .* Array(cₚ) .* Az * Δz

    for i in 1:nz
        for j in 1:nr
            # All Neighbors
            i_neighbors = [i - 1 j; i + 1 j; i j - 1; i j + 1]

            # Remove out-of-bounds neighbors
            i_neighbors = i_neighbors[(i_neighbors[:, 1] .>= 1) .* (i_neighbors[:, 2] .>= 1) .* (i_neighbors[:, 1] .<= nz) .* (i_neighbors[:, 2] .<= nr), :]

            # Get linear index for each actual neighbor
            neighbors = [idx[i_neighbors[i, 1], i_neighbors[i, 2]] for i in eachindex(i_neighbors[:, 1])]

            # Multiply each voxel by the correct area and unit length
            if i - 1 >= 1
                A[idx[i, j], idx[i-1, j]] = Az[idx[i, j]] / Δz
            end

            if i + 1 <= nz
                A[idx[i, j], idx[i+1, j]] = Az[idx[i, j]] / Δz
            end

            if j - 1 >= 1
                A[idx[i, j], idx[i, j-1]] = Arn[idx[i, j]] / Δr
            end

            if j + 1 <= nr
                A[idx[i, j], idx[i, j+1]] = Arp[idx[i, j]] / Δr
            end

            k_CPU = Array(k)
            A[idx[i, j], neighbors] .*= (k_CPU[idx[i, j]] .+ k_CPU[neighbors]) * 0.5 / C[idx[i, j]]

        end
    end

    # Main diagonal subtraction
    A .-= Diagonal(vec(sum(A, dims=2)))
    A = cu(A)

    # External factors (convection and baseplate temperature)
    A += h ./ ρ ./ cₚ/Δz .* -Zp + α/(Δz^2) .* -Zn
    A += (h2*2*nr*Δr) ./ (ρ .* cₚ*(2*nr*Δr*Δr .- Δr*Δr)) .* -Rp

    # Converts power input from kW to kK/s and baseplate temp input from kK to kK/s
    # B_old = hcat(sum((1 ./ (cₚ .* ρ .* Δz)) / (π*(Δr*(nr-buffer))^2) .* power_mask, dims=2), sum((α / Δz^2 .* Zn), dims=2))
    B_old = hcat(sum((1 ./ (cₚ .* ρ .* Δz)) / (π*((Δr * (nr-buffer[2]))^2 - (Δr * buffer[1])^2)) .* power_mask, dims=2), sum((α / Δz^2 .* Zn), dims=2))

    Nu = 2

    A = hcat(A, B_old)
    A = vcat(A, zeros(Nu, nz*nr+Nu))

    B = zeros(nz*nr+Nu, Nu)
    B[end, end] = 1e-1 # MAKE SURE SCALE FACTOR IS CORRECT IN CODE
    B[end-1, end-1] = 1e1 # Scale this up by some factor

    e = [e; zeros(Nu)]

    return A, B, e
end

function horizontal_adjacency(nx, ny) # For forward and backward differencing on edges, central on non-edges
    X = zeros(nx*ny, nx*ny)

    lidx = LinearIndices((nx, ny),)

    # Forward differencing on left edge
    for i in 1:nx
        X[lidx[i, 1], lidx[i, 1+1]] = 1
        X[lidx[i, 1], lidx[i, 1]] = -1
    end

    # Central Differencing
    for i in 1:nx
        for j in 2:(ny-1)
            if j < ny
                X[lidx[i, j], lidx[i, j+1]] = 0.5
            end
            if j > 1
                X[lidx[i, j], lidx[i, j-1]] = -0.5
            end
        end
    end

    # Backwards differencing on right edge
    for i in 1:nx
        X[lidx[i, ny], lidx[i, ny]] = 1
        X[lidx[i, ny], lidx[i, ny-1]] = -1
    end

    return X
end

function r_vector(nz, nr, Δr)
    r = zeros(nz, nr)
    for i in 1:nz
        for j in 1:nr
            r[i, j] = j * Δr
        end
    end

    return vec(r)
end

function surface_voxels(nx, ny, nz)
    voxels = zeros(nx, ny, nz)
    voxels[:, :, 1] .= 1.0

    return vec(voxels)
end

function all_2D_surface_voxels(nx, ny)
    # Create individual vectors for every outside surface of a 2D mesh 
    Xright = zeros(nx, ny) # +x
    Xleft = zeros(nx, ny)  # -x
    Yup = zeros(nx, ny)    # +y
    Ydown = zeros(nx, ny)  # -y

    Xright[:, end] .= 1.0
    Xleft[:, 1] .= 1.0
    Yup[1, :] .= 1.0
    Ydown[end, :] .= 1.0

    return vec(Xright), vec(Xleft), vec(Yup), vec(Ydown)
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

# Laplacian fucntion for use with cubic voxels
function voxel_laplacian(nx, ny, nz)
    A = voxel_adjacency(nx, ny, nz)
    D = voxel_degree(A)
    return D - A
end

# Laplacian function for use with non-cubic voxels
function voxel_laplacian_directional(nx, ny)
    X, Y = voxel_adjacency_directional(nx, ny)
    D = voxel_degree(X)
    E = voxel_degree(Y)

    return D - X, E - Y
end

function voxel_degree(A)
    d = sum(A, dims=2)[:, 1]
    return Diagonal(d)
end

# Voxel adjacency matrix construction best used for cubic voxels
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

# Generate voxel adjacency matrices for each cartesian direction for use with non-cubic voxels
function voxel_adjacency_directional(nx, ny, nz)
    X = zeros(nx * ny * nz, nx * ny * nz)
    Y = zeros(nx * ny * nz, nx * ny * nz)
    Z = zeros(nx * ny * nz, nx * ny * nz)

    idx = LinearIndices((nx, ny, nz))

    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                if i < nx
                    X[idx[i, j, k], idx[i+1, j, k]] = 1
                end
                if i > 1
                    X[idx[i, j, k], idx[i-1, j, k]] = 1
                end
                if j < ny
                    Y[idx[i, j, k], idx[i, j+1, k]] = 1
                end
                if j > 1
                    Y[idx[i, j, k], idx[i, j-1, k]] = 1
                end
                if k < nz
                    Z[idx[i, j, k], idx[i, j, k+1]] = 1
                end
                if k > 1
                    Z[idx[i, j, k], idx[i, j, k-1]] = 1
                end
            end
        end
    end

    return X, Y, Z
end

function voxel_adjacency_directional(nx, ny)
    X = zeros(nx * ny, nx * ny)
    Y = zeros(nx * ny, nx * ny)

    idx = LinearIndices((nx, ny))

    for i in 1:nx
        for j in 1:ny
            if i < nx
                X[idx[i, j], idx[i+1, j]] = 1
            end
            if i > 1
                X[idx[i, j], idx[i-1, j]] = 1
            end
            if j < ny
                Y[idx[i, j], idx[i, j+1]] = 1
            end
            if j > 1
                Y[idx[i, j], idx[i, j-1]] = 1
            end
        end
    end

    return X, Y
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