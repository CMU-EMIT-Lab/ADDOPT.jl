
struct TwoBarLumpedDynamics <: TransferDynamics
    Tₘₐₓ::Vector{Float64}
    h::Float64
    T∞::Float64

    L₀::Float64
    A1::Float64
    A2::Float64
    P1::Float64
    P2::Float64

    ρ::Float64
    cₚ::Float64

    E::Float64
    α::Float64
end

@inline Ns(td::TwoBarLumpedDynamics)::Int = 2
state_min(td::TwoBarLumpedDynamics) = [200.0; 200.0]
state_max(td::TwoBarLumpedDynamics) = td.Tₘₐₓ

function dynamics_function!(td::TwoBarLumpedDynamics, ds::AbstractVector{Ty}, s, t, zi) where {Ty}
    T = s
    ds[1] = td.h * td.P1 * (td.T∞ - T[1]) / (td.A1 * td.cₚ * td.ρ)
    ds[2] = td.h * td.P2 * (td.T∞ - T[2]) / (td.A2 * td.cₚ * td.ρ)
end

function temperature!(td::TwoBarLumpedDynamics, T, s, zi)
    T .= s
end

Nε(td::TwoBarLumpedDynamics) = 9
Nc_ε(td::TwoBarLumpedDynamics) = 12
σy(td::TwoBarLumpedDynamics, T) = 200 * (1 / (1 + exp((T - 500) / 80))) + 20
c_ε_min(td::TwoBarLumpedDynamics) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0]
c_ε_max(td::TwoBarLumpedDynamics) = [0; 0; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; 0; 0]
ε_min(td::TwoBarLumpedDynamics) = [-Inf; -Inf; -Inf; -Inf; 0; 0; 0; 0; 0] # Plastic limits
ε_max(td::TwoBarLumpedDynamics) = [Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf] # Plastic limits
function compatibility_constraint!(td::TwoBarLumpedDynamics, c::AbstractVector{Ty}, εₖ, εₖ₋₁, sₖ) where {Ty}
    εₖ_E1, εₖ_E2, εₖ_P1, εₖ_P2, Δεₖ_P1⁺, Δεₖ_P2⁺, Δεₖ_P1⁻, Δεₖ_P2⁻, bpₖ = εₖ
    εₖ₋₁_E1, εₖ₋₁_E2, εₖ₋₁_P1, εₖ₋₁_P2, Δεₖ₋₁_P1⁺, Δεₖ₋₁_P2⁺, Δεₖ₋₁_P1⁻, Δεₖ₋₁_P2⁻, bpₖ₋₁ = εₖ₋₁
    T1, T2 = sₖ

    E, α = td.E, td.α
    A1, A2 = td.A1, td.A2

    c[1] = ((εₖ_E1 + α * (T1 - 293.15) + εₖ_P1) - (εₖ_E2 + α * (T2 - 293.15) + εₖ_P2)) # Compatibility (strain equality)
    c[2] = (A1 * εₖ_E1 * E + A2 * εₖ_E2 * E) # Equilibrium (force balance)
    c[3] = (σy(td, T1) - εₖ_E1 * E) # Plastic tensile limit 1
    c[4] = (σy(td, T2) - εₖ_E2 * E) # Plastic tensile limit 2
    c[5] = (εₖ_E1 * E + σy(td, T1)) # Plastic compressive limit 1
    c[6] = (εₖ_E2 * E + σy(td, T2)) # Plastic compressive limit 2
    c[7] = bpₖ - (σy(td, T1) - εₖ_E1 * E) * Δεₖ_P1⁺ # Complementarity tensile 1 (no yield unless stress reached)
    c[8] = bpₖ - (σy(td, T2) - εₖ_E2 * E) * Δεₖ_P2⁺ # Complementarity tensile 2 (no yield unless stress reached)
    c[9] = bpₖ - (εₖ_E1 * E + σy(td, T1)) * Δεₖ_P1⁻ # Complementarity compressive 1 (no yield unless stress reached)
    c[10] = bpₖ - (εₖ_E2 * E + σy(td, T2)) * Δεₖ_P2⁻ # Complementarity compressive 2 (no yield unless stress reached)
    c[11] = ((εₖ_P1 - εₖ₋₁_P1) - (Δεₖ_P1⁺ - Δεₖ_P1⁻)) # Definitions of Δε 1
    c[12] = ((εₖ_P2 - εₖ₋₁_P2) - (Δεₖ_P2⁺ - Δεₖ_P2⁻)) # Definitions of Δε 2
end