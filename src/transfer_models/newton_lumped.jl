
struct NewtonLumpedDynamics <: TransferDynamics
    h
    T∞
    m
    cₚ
    Tₘₐₓ
end

function dynamics_function!(td::NewtonLumpedDynamics, ds, s) 
    T = s[1]
    ds[1] = td.h * (td.T∞ - T) / (td.m * td.cₚ)
end

Ns(td::NewtonLumpedDynamics) = 1
state_min(td::NewtonLumpedDynamics) = [td.T∞]
state_max(td::NewtonLumpedDynamics) = [td.Tₘₐₓ]