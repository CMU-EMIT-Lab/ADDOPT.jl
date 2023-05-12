struct PlanarGMAWDynamics <: InputDynamics
    l
    xₙ
    zₙ
    F
end

Nu(id::PlanarGMAWDynamics) = 5 # torch position (x,z), trim, WFS, TS
Nr(id::PlanarGMAWDynamics) = 2 # meltpool root and radius

input_min(id::PlanarGMAWDynamics) = [-Inf; -Inf; 0.5; 0; 0]
input_max(id::PlanarGMAWDynamics) = [Inf; Inf; 1.2; 120.0; 20.0]

state_min(id::PlanarGMAWDynamics) = [0.0; 0.0]
state_max(id::PlanarGMAWDynamics) = [Inf; Inf]

function dynamics_function!(id::PlanarGMAWDynamics, dr, s, r, u)
    N = length(s) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)
    rₘₚ, zₘₚ = r[1], r[2] 

    z̄ₘₚ = zₘₚ
    # Compute steady state meltpool radius and z location
    r̄ₘₚ = WFS > 0 && TS > 0 ? wire_diam * √(WFS/TS) * √(1/2)  : wire_diam
    @. ZW = normpdf((xₙ - xₜ) / l * 2)  * (1 - exp(-m/ (ρ * l^2))) * max(sign(Tₗ*m*cₚ - E), 0) * exp((zₙ / l)*10)#* (zₙ^3)
    ZW_sum = sum(ZW) # logistic(1000*(Tₗ*m*cₚ - E) - 20)
    if ZW_sum > 0 && WFS > 0 && TS > 0
        ZW .*= zₙ
        z̄ₘₚ = sum(ZW) / ZW_sum + l 
    end
        
    dr[1] = γᵣ * (r̄ₘₚ - rₘₚ)
    dr[2] = γₕ * (z̄ₘₚ - zₘₚ)
end

function input_function!(id::PlanarGMAWDynamics, ds, r, u)
    N = length(s) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    F, xₙ, zₙ = id.F, id.xₙ, id.zₙ
    rₘₚ, zₘₚ = r[1], r[2] 
    xₜ, zₜ, I, V, WFS, TS = 0

    ṁ = WFS * π * (wire_diam / 2)^2 * ρ
    P = I * V

    @. F = normpdf((xₙ - xₜ) / l / wₓ) * √(max(rₘₚ^2 - (zₙ - zₘₚ)^2, 0)) * logistic((zₙ - zₘₚ) / l + bₕ)

    if sum(F) > 0
        F ./= sum(F) # Normalize for conservation purposes
    else
        F .= 0
    end

    # Forced / input dynamics
    @. dE += η * F * P              # Add in torch power
    @. dE += F * (cₚ * ṁ * T∞)      # Add in energy contribution from incoming wire (assume room temp)
    @. dm = F * ṁ
end


function ensemble_mass_dynamics!(dm, m, ṁ, F, E, cₚ, dE) 
    # Naive method, assume direct mass transfer to nearest nodes, no flow beyond that
    dm .= F .* ṁ    
end
                

function ensemble_total_dynamics!(dState, state, N,   # Rates and states
                                  xₜ, zₜ, I, V, WFS, TS, lnum,                # Inputs
                                  n_rows, n_cols, l, xₙ, zₙ,                 # Model geometry parameters
                                  k, ρ, cₚ, T∞, T₀, wire_diam,              # Physical constants
                                  h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ,           # Tuning parameters
                                  B, F, ZW, C, K)             # Baseplate mask, torch fraction matrix
    # Set up views
    dE = view(dState, 1:N)
    dm = view(dState, (N+1):2N)
    dpool_params = view(dState, (2N+1):(2N+2))
    E = view(state, 1:N)
    m = view(state, (N+1):2N)
    #pool_params = view(state, (2N+1):(2N+2))
    

    K .= clamp.(E ./ m ./ cₚ, T∞, 3000)
#     temperature!(state, K)
    map!(k, K, K)
    
    # Mass and energy input calculations
    ṁ = WFS * π * (wire_diam / 2)^2 * ρ
    P = I * V
    μ(mi, mj) = min(mi, mj) / mi

    # Get meltpool radius and z-root out of state
    rₘₚ = state[2N+1]
    zₘₚ = state[2N+2]
    
    @. F = normpdf((xₙ - xₜ) / l / wₓ) * √(max(rₘₚ^2 - (zₙ - zₘₚ)^2, 0)) * logistic((zₙ - zₘₚ) / l + bₕ)

    if sum(F) > 0
        F ./= sum(F) # Normalize for conservation purposes
    else
        F .= 0
    end

    z̄ₘₚ = zₘₚ
    # Compute steady state meltpool radius and z location
    r̄ₘₚ = WFS > 0 && TS > 0 ? wire_diam * √(WFS/TS) * √(1/2)  : wire_diam
    @. ZW = normpdf((xₙ - xₜ) / l * 2)  * (1 - exp(-m/ (ρ * l^2))) * max(sign(Tₗ*m*cₚ - E), 0) * exp((zₙ / l)*10)#* (zₙ^3)
    ZW_sum = sum(ZW) # logistic(1000*(Tₗ*m*cₚ - E) - 20)
    if ZW_sum > 0 && WFS > 0 && TS > 0
        ZW .*= zₙ
        z̄ₘₚ = sum(ZW) / ZW_sum + l 
    end
        
    dpool_params[1] = γᵣ * (r̄ₘₚ - rₘₚ)
    dpool_params[2] = γₕ * (z̄ₘₚ - zₘₚ)

    ensemble_energy_dynamics!(dE, E, m, ṁ, P, xₙ, l, n_cols, n_rows, k, ρ, cₚ, T∞, T₀, h∞, h₀, hₐᵣ, η, B, F, μ, xₜ, C, K)
    ensemble_mass_dynamics!(dm, m, ṁ, F, E, cₚ, dE)
end