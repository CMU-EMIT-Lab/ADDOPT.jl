struct HardnessDynamics <: PropertyDynamics
    A
    τ
end

function dynamics_function!(pd::HardnessDynamics, dα, α, s)
    T = s[1]
    Ġ = pd.A * exp(-pd.τ / T)
    dα[1] = (1 - α[1]) * Ġ
end

Nα(pd::HardnessDynamics) = 1
property_min(pd::HardnessDynamics) = [0.0]
property_max(pd::HardnessDynamics) = [1.0]