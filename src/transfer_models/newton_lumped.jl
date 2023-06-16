
struct NewtonLumpedDynamics <: TransferDynamics
    h::Float64
    T∞::Float64
    m::Float64
    cₚ::Float64
    Tₘₐₓ::Float64
end

@inline Ns(td::NewtonLumpedDynamics)::Int = 1
state_min(td::NewtonLumpedDynamics) = [200.0]
state_max(td::NewtonLumpedDynamics) = [td.Tₘₐₓ]

function dynamics_function!(td::NewtonLumpedDynamics, ds::AbstractVector{Ty}, s, t) where Ty
    T = s[1]
    ds[1] = td.h * (td.T∞ - T) / (td.m * td.cₚ)
end

function temperature!(td::NewtonLumpedDynamics, T, s)
    T .= s
end