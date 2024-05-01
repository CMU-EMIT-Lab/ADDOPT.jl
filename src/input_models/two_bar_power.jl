struct TwoBarPowerDynamics <: InputDynamics
    max_power::Vector{Float64}
    min_power::Vector{Float64}
    total_power::Float64    
    h::Float64
    T∞::Float64
    L₀::Float64
    A1::Float64
    A2::Float64
    P1::Float64
    P2::Float64
    ρ::Float64
    cₚ::Float64
end

function dynamics_function!(id::TwoBarPowerDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where Ty

end

function input_function!(id::TwoBarPowerDynamics, ds::AbstractVector{Ty}, r, u, t, zi, Δt) where Ty
    ds[1] += u[1] / (id.L₀ * id.A1 * id.cₚ * id.ρ)
    ds[2] += u[2] / (id.L₀ * id.A2 * id.cₚ * id.ρ)
end

@inline Nu(id::TwoBarPowerDynamics)::Int = 2
@inline Nr(id::TwoBarPowerDynamics)::Int = 0

input_min(id::TwoBarPowerDynamics) = id.min_power
input_max(id::TwoBarPowerDynamics) = id.max_power
input_idle(id::TwoBarPowerDynamics) = [0.0; 0.0]

state_min(id::TwoBarPowerDynamics) = []
state_max(id::TwoBarPowerDynamics) = []

Nc_eq(id::TwoBarPowerDynamics) = 1

function equality_constraint!(id::TwoBarPowerDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    c[1] = u[1] + u[2] - id.total_power
end