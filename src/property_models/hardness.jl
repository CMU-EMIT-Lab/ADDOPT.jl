struct HardnessDynamics <: PropertyDynamics
    A
    τ
end

function dynamics_function!(property_dynamics::HardnessDynamics, dα, α, s)
    T = s[1]
    Ġ = A * exp(-τ / T)
    dα[1] = (1 - α) * Ġ
end

Nα(property_dynamics::HardnessDynamics) = 1
property_min(property_dynamics::HardnessDynamics) = 0.0
property_max(property_dynamics::HardnessDynamics) = 1.0