struct UniformPowerDynamics <: InputDynamics
    max_power
    m
    cₚ
end

function dynamics_function!(input_dynamics::UniformPowerDynamics, dr, r, u)

end

function input_function!(input_dynamics::UniformPowerDynamics, ds, u)
    ds[1] += u[1] / (input_dynamics.m * input_dynamics.cₚ)
end

Nu(input_dynamics::UniformPowerDynamics) = 1
Nr(input_dynamics::UniformPowerDynamics) = 0

input_min(input_dynamics::UniformPowerDynamics) = 0
input_max(input_dynamics::UniformPowerDynamics) = input_dynamics.max_power

state_min(input_dynamics::UniformPowerDynamics) = Inf
state_max(input_dynamics::UniformPowerDynamics) = Inf