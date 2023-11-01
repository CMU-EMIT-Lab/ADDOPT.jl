struct PlanarVoxelTemperatureDynamics <: TransferDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    k
    ρ::Float64
    cₚ::Float64
    T∞::Float64

    h∞::Float64

    Tmax::Float64
    Tmin::Float64

    K_cache::Dict{Tuple{DataType, Int},Any}

    function PlanarVoxelTemperatureDynamics(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin)
        K = Dict{Tuple{DataType, Int},Any}()

        return new(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin, K)#, B, C, K, T)
    end
end

@inline Ns(td::PlanarVoxelTemperatureDynamics)::Int = td.nrows * td.ncols
state_min(td::PlanarVoxelTemperatureDynamics) = td.Tmin * ones(Ns(td))
state_max(td::PlanarVoxelTemperatureDynamics) = td.Tmax * ones(Ns(td)) #Inf * ones(Ns(td))

function dynamics_function!(td::PlanarVoxelTemperatureDynamics, ds::AbstractVector{Ty}, s, t, zi) where {Ty}
    n_rows, n_cols = td.nrows, td.ncols
    l, xₙ, zₙ = td.l, td.xₙ, td.zₙ
    k, ρ, cₚ, T∞ = td.k, td.ρ, td.cₚ, td.T∞
    h∞ = td.h∞

    thread::Int = Threads.threadid()
    K = get!(td.K_cache, (Ty, thread)) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    N = Ns(td)
    dT = ds
    T = s

    map!(k, K, T)

    rcT = reshape(T, (n_cols, n_rows))
    rcdT = reshape(dT, (n_cols, n_rows))
    rcK = reshape(view(K, :), (n_cols, n_rows))

    dT .= 0

    # From top
    dTᵢ = view(rcdT, :, 1:(n_rows-1))
    Tᵢ = view(rcT, :, 1:(n_rows-1))
    Tⱼ = view(rcT, :, 2:n_rows)
    Kᵢ = view(rcK, :, 1:(n_rows-1))
    Kⱼ = view(rcK, :, 2:n_rows)
    dTᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (Tⱼ .- Tᵢ)

    # From bottom
    dTᵢ = view(rcdT, :, 2:n_rows)
    Tᵢ = view(rcT, :, 2:n_rows)
    Tⱼ = view(rcT, :, 1:(n_rows-1))
    Kᵢ = view(rcK, :, 2:n_rows)
    Kⱼ = view(rcK, :, 1:(n_rows-1))
    dTᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (Tⱼ .- Tᵢ)

    # From left
    dTᵢ = view(rcdT, 2:n_cols, :)
    Tᵢ = view(rcT, 2:n_cols, :)
    Tⱼ = view(rcT, 1:(n_cols-1), :)
    Kᵢ = view(rcK, 2:n_cols, :)
    Kⱼ = view(rcK, 1:(n_cols-1), :)
    dTᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (Tⱼ .- Tᵢ)

    # From right
    dTᵢ = view(rcdT, 1:(n_cols-1), :)
    Tᵢ = view(rcT, 1:(n_cols-1), :)
    Tⱼ = view(rcT, 2:n_cols, :)
    Kᵢ = view(rcK, 1:(n_cols-1), :)
    Kⱼ = view(rcK, 2:n_cols, :)
    dTᵢ .+= (1 / (ρ * (l^2) * cₚ)) .* ((Kᵢ .+ Kⱼ) ./ 2) .* (Tⱼ .- Tᵢ)

    dT .+= (h∞ / (ρ * l * cₚ)) .* (T∞ .- T) # Convection to environment

    # Convection to environment out left edge
    dTᵢ = view(rcdT, 1:1, :)
    Tᵢ = view(rcT, 1:1, :)
    dTᵢ .+= (h∞ / (ρ * l * cₚ)) .* (T∞ .- Tᵢ)

    # Convection to environment out right edge
    dTᵢ = view(rcdT, n_cols:n_cols, :)
    Tᵢ = view(rcT, n_cols:n_cols, :)
    dTᵢ .+= (h∞ / (ρ * l * cₚ)) .* (T∞ .- Tᵢ)

    # Convection to environment out top edge
    dTᵢ = view(rcdT, :, 1:1)
    Tᵢ = view(rcT, :, 1:1)
    dTᵢ .+= (h∞ / (ρ * l * cₚ)) .* (T∞ .- Tᵢ)

    # Convection to environment out bottom edge
    dTᵢ = view(rcdT, :, n_rows:n_rows)
    Tᵢ = view(rcT, :, n_rows:n_rows)
    dTᵢ .+= (h∞ / (ρ * l * cₚ)) .* (T∞ .- Tᵢ)
end

function temperature!(td::PlanarVoxelTemperatureDynamics, T, s, zi)
    T .= s
end