include("input_models/uniform_power.jl")

include("property_models/hardness.jl")

include("transfer_models/newton_lumped.jl")

struct Process
    input_dynamics::InputDynamics
    transfer_dynamics::TransferDynamics
    property_dynamics::PropertyDynamics
end

function Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)
   id = UniformPowerDynamics(Pₘₐₓ, m, cₚ)
   td = NewtonLumpedDynamics(h, T∞, m, cₚ, Tₘₐₓ)
   pd = HardnessDynamics(1e4, 1e4)
   
   return Process(id, td, pd)
end