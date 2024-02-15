struct PlanarVoxelTemperatureDynamics <: TransferDynamics
    nrows::Int
    ncols::Int
    ndeep::Int

    l::Float64
    lz::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    k::Float64
    ρ::Float64
    cₚ::Float64
    T∞::Float64

    h∞::Float64

    Tmax::Vector{Float64}
    Tmin::Vector{Float64}

    function PlanarVoxelTemperatureDynamics(nrows, ncols, ndeep, l, lz, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin)

        if isa(Tmax, Number)
            Tmax = Tmax .* ones(nrows * ncols * ndeep)
        end

        if isa(Tmin, Number)
            Tmin = Tmin .* ones(nrows * ncols * ndeep)
        end

        return new(nrows, ncols, ndeep, l, lz, xₙ, zₙ, k, ρ, cₚ, T∞, h∞, Tmax, Tmin)
    end
end

@inline Ns(td::PlanarVoxelTemperatureDynamics)::Int = td.nrows * td.ncols * td.ndeep
state_min(td::PlanarVoxelTemperatureDynamics) = td.Tmin
state_max(td::PlanarVoxelTemperatureDynamics) = td.Tmax

function dynamics_function!(td::PlanarVoxelTemperatureDynamics, ds::AbstractVector{Ty}, s, t, zi) where {Ty}
    nrows, ncols, ndeep = td.nrows, td.ncols, td.ndeep
    l, lz, xₙ, zₙ = td.l, td.lz, td.xₙ, td.zₙ
    k, ρ, cₚ, T∞ = td.k, td.ρ, td.cₚ, td.T∞
    h∞ = td.h∞

    α = k / (ρ * cₚ)

    dT = ds
    T = s

    dT .= 0
    rcT = reshape(T, (ncols, nrows, ndeep))
    rcdT = reshape(dT, (ncols, nrows, ndeep))

    # Down
    dTᵢ = view(rcdT, :, :, 1:(ndeep-1))
    Tᵢ = view(rcT, :, :, 1:(ndeep-1))
    Tⱼ = view(rcT, :, :, 2:ndeep)
    dTᵢ .+= (α / lz^2) .* (Tⱼ .- Tᵢ)

    # Up
    dTᵢ = view(rcdT, :, :, 2:ndeep)
    Tᵢ = view(rcT, :, :, 2:ndeep)
    Tⱼ = view(rcT, :, :, 1:(ndeep-1))
    dTᵢ .+= (α / lz^2) .* (Tⱼ .- Tᵢ)

    # From top
    dTᵢ = view(rcdT, :, 1:(nrows-1), :)
    Tᵢ = view(rcT, :, 1:(nrows-1), :)
    Tⱼ = view(rcT, :, 2:nrows, :)
    dTᵢ .+= (α / l^2) .* (Tⱼ .- Tᵢ)

    # From bottom
    dTᵢ = view(rcdT, :, 2:nrows, :)
    Tᵢ = view(rcT, :, 2:nrows, :)
    Tⱼ = view(rcT, :, 1:(nrows-1), :)
    dTᵢ .+= (α / l^2) .* (Tⱼ .- Tᵢ)

    # From left
    dTᵢ = view(rcdT, 2:ncols, :, :)
    Tᵢ = view(rcT, 2:ncols, :, :)
    Tⱼ = view(rcT, 1:(ncols-1), :, :)
    dTᵢ .+= (α / l^2) .* (Tⱼ .- Tᵢ)

    # From right
    dTᵢ = view(rcdT, 1:(ncols-1), :, :)
    Tᵢ = view(rcT, 1:(ncols-1), :, :)
    Tⱼ = view(rcT, 2:ncols, :, :)
    dTᵢ .+= (α / l^2) .* (Tⱼ .- Tᵢ)

    # Convection to environemnt from up face
    dTᵢ = view(rcdT, :, :, 1:1)
    Tᵢ = view(rcT, :, :, 1:1)
    dTᵢ .+= (h∞ / (ρ * lz * cₚ)) .* (T∞ .- Tᵢ)

    # Conduction to environment from down face
    dTᵢ = view(rcdT, :, :, ndeep:ndeep)
    Tᵢ = view(rcT, :, :, ndeep:ndeep)
    dTᵢ .+= (α / lz^2) .* (T∞ .- Tᵢ)

    # Conduction to environment out left edge
    dTᵢ = view(rcdT, 1:1, :, :)
    Tᵢ = view(rcT, 1:1, :, :)
    dTᵢ .+= (α / l^2) .* (T∞ .- Tᵢ)

    # Conduction to environment out right edge
    dTᵢ = view(rcdT, ncols:ncols, :, :)
    Tᵢ = view(rcT, ncols:ncols, :, :)
    dTᵢ .+= (α / l^2) .* (T∞ .- Tᵢ)

    # Conduction to environment out top edge
    dTᵢ = view(rcdT, :, 1:1, :)
    Tᵢ = view(rcT, :, 1:1, :)
    dTᵢ .+= (α / l^2) .* (T∞ .- Tᵢ)

    # Conduction to environment out bottom edge
    dTᵢ = view(rcdT, :, nrows:nrows, :)
    Tᵢ = view(rcT, :, nrows:nrows, :)
    dTᵢ .+= (α / l^2) .* (T∞ .- Tᵢ)
end

function temperature!(td::PlanarVoxelTemperatureDynamics, T, s, zi)
    T .= s
end