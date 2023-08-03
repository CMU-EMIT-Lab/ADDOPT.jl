using StatsFuns
using Interpolations
using Symbolics

struct NullInputDynamics <: InputDynamics
end

Nu(id::NullInputDynamics)::Int = 1
Nr(id::NullInputDynamics)::Int = 0

input_min(id::NullInputDynamics) = [0.0]
input_max(id::NullInputDynamics) = [0.0]
input_idle(id::NullInputDynamics) = [0.0]

state_min(id::NullInputDynamics) = []
state_max(id::NullInputDynamics) = []

function dynamics_function!(id::NullInputDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}
end

function input_function!(id::NullInputDynamics, ds::AbstractVector{Ty}, r, u, t, zi) where {Ty}
end