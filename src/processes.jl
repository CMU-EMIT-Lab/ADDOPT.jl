# Input models
include("input_models/uniform_power.jl")
include("input_models/planar_gmaw.jl")

# Property models
include("property_models/hardness.jl")

# Transfer models
include("transfer_models/newton_lumped.jl")
include("transfer_models/planar_voxel_mass_temp.jl")

struct Process{ID<:InputDynamics, TD<:TransferDynamics, PD<:PropertyDynamics}
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

function PlanarWAAMHardness(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, A, τ)
    n_voxels = nrows * ncols
    id = PlanarGMAWDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
    td = PlanarVoxelMassEnergyDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀)
    pd = HardnessDynamics(A, τ, n=n_voxels)

    return Process(id, td, pd)
end