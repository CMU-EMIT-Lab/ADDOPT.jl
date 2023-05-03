struct UniformPowerDynamics <: InputDynamics
    max_power
    m
    cₚ
end

function dynamics_function!(id::UniformPowerDynamics, dr, s, r, u)

end

function input_function!(id::UniformPowerDynamics, ds, r, u)
    ds[1] += u[1] / (id.m * id.cₚ)
end

Nu(id::UniformPowerDynamics) = 1
Nr(id::UniformPowerDynamics) = 0

input_min(id::UniformPowerDynamics) = [0.0]
input_max(id::UniformPowerDynamics) = [id.max_power]

state_min(id::UniformPowerDynamics) = [Inf]
state_max(id::UniformPowerDynamics) = [Inf]