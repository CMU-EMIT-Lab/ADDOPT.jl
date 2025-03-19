
include("integration.jl")
include("voxelized_conduction.jl")
include("bead_generation.jl")
include("finite_diff.jl")

function refine_grid(u, n)
    nx, ny = size(u)
    nnx, nny = n * nx, n * ny

    un = zeros(eltype(u), (nnx, nny))

    for i in 1:nx
        for j in 1:ny
            un[(1:n).+(n*(i-1)), (1:n).+(n*(j-1))] .= u[i, j]
        end
    end

    return un
end