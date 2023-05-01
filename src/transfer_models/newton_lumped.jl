
struct NewtonLumpedDynamics <: TransferDynamics
    h
    T∞
    m
    cₚ
    Tₘₐₓ
end

function dynamics_function!(transfer_dynamics::NewtonLumpedDynamics, ds, s) 
    T = s[1]
    ds[1] = h * (transfer_dynamics.T∞ - T) / (transfer_dynamics.m * transfer_dynamics.cₚ)
end

Ns(transfer_dynamics::NewtonLumpedDynamics) = 1
state_min(transfer_dynamics::NewtonLumpedDynamics) = 0
state_max(transfer_dynamics::NewtonLumpedDynamics) = Tₘₐₓ