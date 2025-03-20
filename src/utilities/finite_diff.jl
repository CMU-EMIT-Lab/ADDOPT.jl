function finite_diff!(T_2, T_1, mask, P, l, dt, k, ρ, cₚ, h∞, T∞)
    cidxs = CartesianIndices(T_1)
    Nx, Ny, Nz = size(T_1)

    T_2 .= P

    for (cidx, T) in zip(cidxs, T_1)
        if !mask[cidx]
            continue
        end
        x, y, z = cidx[1], cidx[2], cidx[3]

        if x > 1 && mask[x-1, y, z]
            T_2[cidx] += (T_1[x-1, y, z] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l^2 * h∞
        end

        if x < Nx && mask[x+1, y, z]
            T_2[cidx] += (T_1[x+1, y, z] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l^2 * h∞
        end

        if y > 1 && mask[x, y-1, z]
            T_2[cidx] += (T_1[x, y-1, z] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l^2 * h∞
        end

        if y < Ny && mask[x, y+1, z]
            T_2[cidx] += (T_1[x, y+1, z] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l^2 * h∞
        end

        if z > 1 && mask[x, y, z-1]
            T_2[cidx] += (T_1[x, y, z-1] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l * k
        end

        if z < Nz && mask[x, y, z+1]
            T_2[cidx] += (T_1[x, y, z+1] - T) * l * k
        else
            T_2[cidx] += (T∞ - T) * l^2 * h∞
        end
    end

    T_2 .*= dt / (ρ * cₚ * l^3)
    T_2 .+= T_1
end

function finite_diff!(T_2::CuArray{T,N,M}, T_1::CuArray{T,N,M}, mask, P, l, dt, k, ρ, cₚ, h∞, T∞) where {T,N,M}
    T_2 .= P

    Nx, Ny, Nz = size(T_1)

    block_x = ceil(Int, Nx / 8)
    block_y = ceil(Int, Ny / 8)
    block_z = ceil(Int, Nz / 8)

    @cuda threads = (8, 8, 8) blocks = (block_x, block_y, block_z) heat_flow_finite_diff_kernel(T_2, T_1, mask, l, k, h∞, T∞)

    T_2 .*= dt / (ρ * cₚ * l^3)
    T_2 .+= T_1
end

function heat_flow_finite_diff_kernel(T_2, T_1, mask, l, k, h∞, T∞)
    Nx, Ny, Nz = size(T_1)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    z = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if x > Nx || y > Ny || z > Nz || !mask[x, y, z]
        return nothing
    end

    @inbounds T = T_1[x, y, z]

    if x > 1 && mask[x-1, y, z]
        @inbounds T_2[x, y, z] += (T_1[x-1, y, z] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l^2 * h∞
    end

    if x < Nx && mask[x+1, y, z]
        @inbounds T_2[x, y, z] += (T_1[x+1, y, z] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l^2 * h∞
    end

    if y > 1 && mask[x, y-1, z]
        @inbounds T_2[x, y, z] += (T_1[x, y-1, z] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l^2 * h∞
    end

    if y < Ny && mask[x, y+1, z]
        @inbounds T_2[x, y, z] += (T_1[x, y+1, z] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l^2 * h∞
    end

    if z > 1 && mask[x, y, z-1]
        @inbounds T_2[x, y, z] += (T_1[x, y, z-1] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l * k
    end

    if z < Nz && mask[x, y, z+1]
        @inbounds T_2[x, y, z] += (T_1[x, y, z+1] - T) * l * k
    else
        @inbounds T_2[x, y, z] += (T∞ - T) * l^2 * h∞
    end

    return nothing
end
