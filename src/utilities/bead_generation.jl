function path_to_points(path, ds, Nk)
    points = zeros(size(path, 1), Nk)

    s = 0.0
    i = 1

    p0 = SVector{3}(path[1, i], path[2, i], path[3, i])
    p1 = SVector{3}(path[1, i+1], path[2, i+1], path[3, i+1])
    dp = p1 - p0
    L = norm(dp)

    for k in 1:Nk
        if s ≥ L
            i += 1
            s -= L

            p0 = SVector{3}(path[1, i], path[2, i], path[3, i])
            p1 = SVector{3}(path[1, i+1], path[2, i+1], path[3, i+1])
            dp = p1 - p0

            L = norm(dp)
        end

        point = p0 + (s / L) * dp
        points[:, k] .= point

        s += ds
    end

    return points
end

function gaussian_3d(x, y, z, px::T, py::T, pz::T, σ::T) where {T}
    return T(exp(-0.5 * (((x - px) / σ)^2 + ((y - py) / σ)^2 + ((z - pz) / σ)^2)) / √((2π)^3) / σ^3)
end

function gaussian_3d!(mesh, px, py, pz, σ)
    map!((cidx) -> begin
            x, y, z = cidx[1], cidx[2], cidx[3]
            return gaussian_3d(x, y, z, px, py, pz, σ)
        end, mesh, CartesianIndices(mesh))
end

function expand_bead!(T, mask, Tₗ)
    map!((T, m) -> (m && T == 0) ? Tₗ : T, T, T, mask)
end

function expand_bead!(Tarr::CuArray{T,N,M}, mask, Tₗ) where {T, N, M}
    nblocks = ceil(Int, length(Tarr) / 256)

    @cuda threads = 256 blocks = nblocks expand_bead_kernel(Tarr, mask, Tₗ)
end

function expand_bead_kernel(T, mask, Tₗ)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    
    if i > length(T)
        return nothing
    end

    if mask[i] && (T[i] == 0)
        T[i] = Tₗ
    end

    return nothing
end