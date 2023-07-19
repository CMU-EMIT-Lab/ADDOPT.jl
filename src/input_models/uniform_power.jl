struct UniformPowerDynamics <: InputDynamics
    max_power::Float64
    m::Float64
    cₚ::Float64
end

function dynamics_function!(id::UniformPowerDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where Ty

end

function input_function!(id::UniformPowerDynamics, ds::AbstractVector{Ty}, r, u, t, zi) where Ty
    ds[1] += u[1] / (id.m * id.cₚ)
end

@inline Nu(id::UniformPowerDynamics)::Int = 1
@inline Nr(id::UniformPowerDynamics)::Int = 0

input_min(id::UniformPowerDynamics) = [200.0]
input_max(id::UniformPowerDynamics) = [id.max_power]
input_idle(id::UniformPowerDynamics) = [0.0]

state_min(id::UniformPowerDynamics) = []
state_max(id::UniformPowerDynamics) = []