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
state_min(td::NewtonLumpedDynamics) = [200.0]
state_max(td::NewtonLumpedDynamics) = [td.Tₘₐₓ]


"""
Adj is the adjacency matrix
x is the array of (x,z) locations of agents
T is the vector of current temperatures (state vector)
xₜ is the (x, z) location of the torch
I, V, WFS, TS are input process parameters
"""
function ensemble_energy_dynamics!(dE, E, m,               # State, derivative, mass
                                    ṁ, P, xₙ,                  # Inputs
                                    l, n_cols, n_rows,                    # Model size parameters
                                    k, ρ, cₚ, T∞, T₀,      # Physical constants
                                    h∞, h₀, hₐᵣ, η,             # Tuning parameters (convection, baseplate resistance, efficiency)
                                    B, F, μ, xₜ, C, K)         # Mass-variant matrices for conduction and convection 
                                                           # + baseplate matrix + fraction torch input matrix
   
    N = length(E)
    rcE = reshape(E, (n_cols, n_rows))
    rcm = reshape(m, (n_cols, n_rows))
    rcdE = reshape(dE, (n_cols, n_rows))
    rcC = reshape(view(C, :), (n_cols, n_rows))
    rcK = reshape(view(K, :), (n_cols, n_rows))

    # Autonomous dynamics (Conduction among agents, convection to environment, conduction through baseplate)    
    dE .= 0
    @. C = 4 / (ρ * l) + l^2 / m * (1 - exp(-m / (ρ * l^2) * 2000)) 
    
    # From top
    dEᵢ = view(rcdE, :, 1:(n_rows-1))
    Eᵢ = view(rcE, :, 1:(n_rows-1))
    Eⱼ = view(rcE, :, 2:n_rows)
    mᵢ = view(rcm, :, 1:(n_rows-1))
    mⱼ = view(rcm, :, 2:n_rows)
    Cᵢ = view(rcC, :, 1:(n_rows-1))
    Kᵢ = view(rcK, :, 1:(n_rows-1))
    Kⱼ = view(rcK, :, 2:n_rows)
    dEᵢ .+= (1 / (ρ*(l^2)*cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ,mᵢ).*Eⱼ .- μ.(mᵢ,mⱼ).*Eᵢ)
    Cᵢ .-= μ.(mᵢ,mⱼ) ./ (ρ * l)
    
    # From bottom
    dEᵢ = view(rcdE, :, 2:n_rows)
    Eᵢ = view(rcE, :, 2:n_rows)
    Eⱼ = view(rcE, :, 1:(n_rows-1))
    mᵢ = view(rcm, :, 2:n_rows)
    mⱼ = view(rcm, :, 1:(n_rows-1))
    Cᵢ = view(rcC, :, 2:n_rows)
    Kᵢ = view(rcK, :, 2:n_rows)
    Kⱼ = view(rcK, :, 1:(n_rows-1))
    dEᵢ .+= (1 / (ρ*(l^2)*cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ,mᵢ).*Eⱼ .- μ.(mᵢ,mⱼ).*Eᵢ)
    Cᵢ .-= μ.(mᵢ,mⱼ) ./ (ρ * l)
    
    # From left
    dEᵢ = view(rcdE, 2:n_cols, :)
    Eᵢ = view(rcE, 2:n_cols, :)
    Eⱼ = view(rcE, 1:(n_cols-1), :)
    mᵢ = view(rcm, 2:n_cols, :)
    mⱼ = view(rcm, 1:(n_cols-1), :)
    Cᵢ = view(rcC, 2:n_cols, :)
    Kᵢ = view(rcK, 2:n_cols, :)
    Kⱼ = view(rcK, 1:(n_cols-1), :)
    dEᵢ .+= (1 / (ρ*(l^2)*cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ,mᵢ).*Eⱼ .- μ.(mᵢ,mⱼ).*Eᵢ)
    Cᵢ .-= μ.(mᵢ,mⱼ) ./ (ρ * l)
    
    # From right
    dEᵢ = view(rcdE, 1:(n_cols-1), :)
    Eᵢ = view(rcE, 1:(n_cols-1), :)
    Eⱼ = view(rcE, 2:n_cols, :)
    mᵢ = view(rcm, 1:(n_cols-1), :)
    mⱼ = view(rcm, 2:n_cols, :)
    Cᵢ = view(rcC, 1:(n_cols-1), :)
    Kᵢ = view(rcK, 1:(n_cols-1), :)
    Kⱼ = view(rcK, 2:n_cols, :)
    dEᵢ .+= (1 / (ρ*(l^2)*cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ,mᵢ).*Eⱼ .- μ.(mᵢ,mⱼ).*Eᵢ)
    Cᵢ .-= μ.(mᵢ,mⱼ) ./ (ρ * l)
    
    dE .+= (h∞ / cₚ) .* C .* (cₚ.*T∞.*m .- E) # Convection to environment
    if ṁ > 0 && P > 0; dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ.*T∞.*m .- E); end # Convection from argon
    dE .+= (h₀ / (ρ*l*cₚ)).*B.*(cₚ.*T₀.*m .- E) # Conduction to baseplate
        
    # Forced / input dynamics
    dE .+= η .* F .* P              # Add in torch power
    dE .+= F .* (cₚ .* ṁ .* T∞)     # Add in energy contribution from incoming wire (assume room temp)
end
        
