struct NullPropertyDynamics <: PropertyDynamics

end

@inline Nα(pd::NullPropertyDynamics)::Int = 0
property_min(pd::NullPropertyDynamics) = []
property_max(pd::NullPropertyDynamics) = []

function dynamics_function!(pd::NullPropertyDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t) where Ty

end