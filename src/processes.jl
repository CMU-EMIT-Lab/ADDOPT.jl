# Input models
include("input_models/uniform_power.jl")
include("input_models/planar_gmaw.jl")
include("input_models/planar_gmaw_prescribed.jl")
include("input_models/planar_heatsource.jl")
include("input_models/planar_heatsource_prescribed.jl")

# Property models
include("property_models/hardness.jl")
include("property_models/fusion.jl")
include("property_models/null_property.jl")

# Transfer models
include("transfer_models/newton_lumped.jl")
include("transfer_models/planar_voxel_mass_temp.jl")
include("transfer_models/planar_voxel_temp.jl")

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

function row_col(n_rows, n_cols, index)
    row = (index - 1) ÷ n_cols + 1
    col = mod(index, 1:n_cols)

    return row, col
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
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀)
    pd = HardnessDynamics(A, τ, n=n_voxels)

    return Process(id, td, pd)
end

function PlanarWAAM(nrows, ncols, l, k, ρ, cₚ, T∞, T₀, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function PlanarWAAMPrescribedMotion(nrows, ncols, l, k, ρ, cₚ, T∞, T₀, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, xmin, xmax, tmin, tmax)
    n_voxels = nrows * ncols

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarGMAWDynamicsPrescribed(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, xmin, xmax, tmin, tmax)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀)
    pd = NullPropertyDynamics()

    return Process(id, td, pd)
end

function PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ)
    n_voxels = nrows * ncols
    A = 1e3
    τ = 5e3

    # Generate x-z matrix/vector
    rc(idx) = row_col(nrows, ncols, idx)
    x = rc.(1:n_voxels)
    x = l .* vcat(collect.(x)'...)
    reverse!(x, dims=2)

    xₙ = x[:, 1]
    zₙ = x[:, 2]

    id = PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, ρ, cₚ, σ)
    td = PlanarVoxelTemperatureDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞)
    pd = FusionDynamics(A, τ; n=n_voxels)

    return Process(id, td, pd)
end

function PlanarLPBFPrescribedMotion(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, xmin, xmax, zmin, zmax, tmin, tmax)
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
    td = PlanarVoxelTemperatureDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞)
    pd = HardnessDynamics(A, τ, n=n_voxels)#FusionDynamics(A, τ; n=n_voxels)

    return Process(id, td, pd)
end