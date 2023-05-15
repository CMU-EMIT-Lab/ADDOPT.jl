struct PlanarVoxelMassEnergyDynamics <: TransferDynamics
    nrows
    ncols

    l
    xₙ
    zₙ

    k
    ρ
    cₚ
    T∞
    wire_diam

    h∞
    h₀

    B
    C
    K
    T

    function PlanarVoxelMassEnergyDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀)
        B = zeros(nrows*ncols)
        C = zeros(nrows*ncols)
        K = zeros(nrows*ncols)
        T = zeros(nrows*ncols)
        
        return new(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, B, C, K, T)
    end
end

Ns(td::PlanarVoxelMassEnergyDynamics) = td.nrows * td.ncols * 2
state_min(td::PlanarVoxelMassEnergyDynamics) = zeros(Ns(td))
state_max(td::PlanarVoxelMassEnergyDynamics) = Inf * ones(Ns(td))


function dynamics_function!(td::PlanarVoxelMassEnergyDynamics, ds, s)
    n_rows, n_cols = td.nrows, td.ncols
    l, xₙ, zₙ = td.l, td.xₙ, td.zₙ
    k, ρ, cₚ, T∞, T₀, wire_diam = td.k, td.ρ, td.cₚ, td.T∞, td.T₀, td.wire_diam
    h∞, h₀ = td.h∞, td.h₀#, td.hₐᵣ, td.η, td.γᵣ, td.γₕ, td.wₓ, td.bₕ
    B, C, K, T = td.B, td.C, td.K, td.T

    μ(mi, mj) = min(mi, mj) / mi

    temperature!(td, T, s)
    map!(k, K, T)

    N = Ns(td) ÷ 2
    dE = view(ds, 1:N)
    dm = view(ds, (N+1):2N)
    E = view(s, 1:N)
    m = view(s, (N+1):2N)

    rcE = reshape(E, (n_cols, n_rows))
    rcm = reshape(m, (n_cols, n_rows))
    rcdE = reshape(dE, (n_cols, n_rows))
    rcC = reshape(view(C, :), (n_cols, n_rows))
    rcK = reshape(view(K, :), (n_cols, n_rows))

    dm .= 0
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
    dEᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ, mᵢ) .* Eⱼ .- μ.(mᵢ, mⱼ) .* Eᵢ)
    Cᵢ .-= μ.(mᵢ, mⱼ) ./ (ρ * l)

    # From bottom
    dEᵢ = view(rcdE, :, 2:n_rows)
    Eᵢ = view(rcE, :, 2:n_rows)
    Eⱼ = view(rcE, :, 1:(n_rows-1))
    mᵢ = view(rcm, :, 2:n_rows)
    mⱼ = view(rcm, :, 1:(n_rows-1))
    Cᵢ = view(rcC, :, 2:n_rows)
    Kᵢ = view(rcK, :, 2:n_rows)
    Kⱼ = view(rcK, :, 1:(n_rows-1))
    dEᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ, mᵢ) .* Eⱼ .- μ.(mᵢ, mⱼ) .* Eᵢ)
    Cᵢ .-= μ.(mᵢ, mⱼ) ./ (ρ * l)

    # From left
    dEᵢ = view(rcdE, 2:n_cols, :)
    Eᵢ = view(rcE, 2:n_cols, :)
    Eⱼ = view(rcE, 1:(n_cols-1), :)
    mᵢ = view(rcm, 2:n_cols, :)
    mⱼ = view(rcm, 1:(n_cols-1), :)
    Cᵢ = view(rcC, 2:n_cols, :)
    Kᵢ = view(rcK, 2:n_cols, :)
    Kⱼ = view(rcK, 1:(n_cols-1), :)
    dEᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ, mᵢ) .* Eⱼ .- μ.(mᵢ, mⱼ) .* Eᵢ)
    Cᵢ .-= μ.(mᵢ, mⱼ) ./ (ρ * l)

    # From right
    dEᵢ = view(rcdE, 1:(n_cols-1), :)
    Eᵢ = view(rcE, 1:(n_cols-1), :)
    Eⱼ = view(rcE, 2:n_cols, :)
    mᵢ = view(rcm, 1:(n_cols-1), :)
    mⱼ = view(rcm, 2:n_cols, :)
    Cᵢ = view(rcC, 1:(n_cols-1), :)
    Kᵢ = view(rcK, 1:(n_cols-1), :)
    Kⱼ = view(rcK, 2:n_cols, :)
    dEᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (μ.(mⱼ, mᵢ) .* Eⱼ .- μ.(mᵢ, mⱼ) .* Eᵢ)
    Cᵢ .-= μ.(mᵢ, mⱼ) ./ (ρ * l)

    dE .+= (h∞ / cₚ) .* C .* (cₚ .* T∞ .* m .- E) # Convection to environment

    # if ṁ > 0 && P > 0
    #     dE .+= (hₐᵣ / cₚ) .* C .* normpdf.((xₙ .- xₜ) ./ 0.006) .* (cₚ .* T∞ .* m .- E)
    # end # Convection from argon
    
    dE .+= (h₀ / (ρ * l * cₚ)) .* B .* (cₚ .* T₀ .* m .- E) # Conduction to baseplate
end

function temperature!(td::PlanarVoxelMassEnergyDynamics, T, s)
    N = Ns(td) ÷ 2
    E = view(s, 1:N)
    m = view(s, (N+1):2N)

    T .= clamp.(E ./ m ./ td.cₚ, td.T∞, 3000)
end