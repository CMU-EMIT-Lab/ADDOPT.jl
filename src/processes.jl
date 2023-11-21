# Input models
include("input_models/uniform_power.jl")
include("input_models/planar_gmaw.jl")
include("input_models/planar_gmaw_prescribed.jl")
# include("input_models/planar_heatsource.jl")
include("input_models/planar_heatsource_relaxed.jl")
include("input_models/planar_heatsource_prescribed.jl")
include("input_models/gmaw_prescribed.jl")
include("input_models/gmaw_fully_prescribed.jl")
include("input_models/null_input.jl")

# Property models
include("property_models/hardness.jl")
include("property_models/fusion.jl")
include("property_models/null_property.jl")

# Transfer models
include("transfer_models/newton_lumped.jl")
include("transfer_models/planar_voxel_mass_temp.jl")
include("transfer_models/planar_voxel_temp.jl")
include("transfer_models/voxel_energy_fill.jl")
include("transfer_models/voxel_energy.jl")
include("transfer_models/voxel_prescribed_temp.jl")

struct Process{ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics}
    input_dynamics::ID
    transfer_dynamics::TD
    property_dynamics::PD
end

function Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)
    id = UniformPowerDynamics(Pₘₐₓ, m, cₚ)
    td = NewtonLumpedDynamics(h, T∞, m, cₚ, Tₘₐₓ)
    pd = HardnessDynamics(1e4, 9625)

    return Process(id, td, pd)
end

function FurnaceSimple(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)
    id = UniformPowerDynamics(Pₘₐₓ, m, cₚ)
    td = NewtonLumpedDynamics(h, T∞, m, cₚ, Tₘₐₓ)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function row_col(n_rows, n_cols, index)
    row = (index - 1) ÷ n_cols + 1
    col = mod(index, 1:n_cols)

    return row, col
end

function gen_xyz(nx, ny, nz, l)
    x = zeros(nx, ny, nz)
    y = zeros(nx, ny, nz)
    z = zeros(nx, ny, nz)

    for i in 1:nx
        x[i, :, :] .= i
    end

    for i in 1:ny
        y[:, i, :] .= i
    end

    for i in 1:nz
        z[:, :, i] .= i
    end

    x = reshape(x, (nx * ny * nz)) .* l
    y = reshape(y, (nx * ny * nz)) .* l
    z = reshape(z, (nx * ny * nz)) .* l

    return x, y, z
end

function WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, r_bead, Δr, xfill, xtorch)
    xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)
    Tₗ = 1700.0 # K

    id = GMAWDynamicsFullyPrescribed(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, r_bead, Δr, p̄, xtorch)
    td = VoxelEnergyDynamics(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, xfill)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function WAAMHardnessPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ, r_bead, Δr, xfill, xtorch)
    xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)
    n_voxels = nx * ny * nz
    Tₗ = 1700.0 # K

    id = GMAWDynamicsFullyPrescribed(nx, ny, nz, l, xₙ, yₙ, zₙ, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, r_bead, Δr, p̄, xtorch)
    td = VoxelEnergyDynamics(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, xfill)
    pd = HardnessDynamics(A, τ, n=n_voxels)

    return Process(id, td, pd)
end

function WAAMHardnessPrescribedTemp(nvox, Y, A1, τ1, A2, τ2)
    Tₗ = 1700.0 # K

    id = NullInputDynamics()
    td = VoxelPrescribedTemperatureDynamics(Y)
    pd = HardnessDynamics(A1, τ1, A2, τ2, n=nvox)

    return Process(id, td, pd)
end

function WAAMHardnessCooling(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ, x)
    xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)
    n_voxels = nx * ny * nz
    Tₗ = 1700.0 # K

    id = NullInputDynamics()
    td = VoxelEnergyDynamics(nx, ny, nz, l, xₙ, yₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax, x)
    pd = HardnessDynamics(A, τ, n=n_voxels)

    return Process(id, td, pd)
end

function PlanarWAAMHardness(nrows, ncols, l, k, ρ, cₚ, T∞, T₀, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, A, τ)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
    pd = HardnessDynamics(A, τ, n=n_voxels)

    return Process(id, td, pd)
end

function PlanarWAAM(nrows, ncols, l, w, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, Tmin, Tmax)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function PlanarWAAMPrescribedMotion(nrows, ncols, l, w, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, Tmin, Tmax, xmin, xmax, tmin, tmax)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarGMAWDynamicsPrescribed(nrows, ncols, l, xₙ, zₙ, ρₘ, cₚₘ, T∞, 1700, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, xmin, xmax, tmin, tmax)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, w, xₙ, zₙ, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, Tmin, Tmax)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, Pₘₐₓ, Pₘᵢₙ, vₘₐₓ, Tmelt, τ, Tmax, Tmin)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, ρ, cₚ, σ, vₘₐₓ)
    td = PlanarVoxelTemperatureDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin)
    # pd = FusionDynamics(Tmelt, τ; n=n_voxels)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function PlanarLPBFPrescribedMotion(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, xmin, xmax, zmin, zmax, tmin, tmax, Tmax, Tmin)
    n_voxels = nrows * ncols
    A = 1e4
    τ = 9625

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarHeatsourcePrescribedMotionDynamics(nrows, ncols, l, xₙ, zₙ, ρ, cₚ, σ, xmin, xmax, zmin, zmax, tmin, tmax)
    td = PlanarVoxelTemperatureDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin)
    pd = HardnessDynamics(A, τ, n=n_voxels)#FusionDynamics(A, τ; n=n_voxels)

    return Process(id, td, pd)
end