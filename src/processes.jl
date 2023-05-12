# Input models
include("input_models/uniform_power.jl")
include("input_models/planar_gmaw.jl")

# Property models
include("property_models/hardness.jl")

# Transfer models
include("transfer_models/newton_lumped.jl")
include("transfer_models/planar_voxel_mass_temp.jl")

struct Process
    input_dynamics::InputDynamics
    transfer_dynamics::TransferDynamics
    property_dynamics::PropertyDynamics
end

function Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)
   id = UniformPowerDynamics(Pₘₐₓ, m, cₚ)
   td = NewtonLumpedDynamics(h, T∞, m, cₚ, Tₘₐₓ)
   pd = HardnessDynamics(1e4, 9625)
   
   return Process(id, td, pd)
end

function PlanarWAAMHardness()
    id = PlanarGMAWDynamics()
    td = 
    pd = HardnessDynamics

    return Process(id, td, pd)
end