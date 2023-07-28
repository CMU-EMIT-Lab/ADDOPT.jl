using StatsFuns

struct PlanarHeatsourceDynamics <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    σ::Float64

    ρ::Float64
    cₚ::Float64

    F_cache::Dict{Tuple{DataType, Int},Any}

    function PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, ρ, cₚ, σ)
        F = Dict{Tuple{DataType, Int},Any}()

        return new(nrows, ncols, l, xₙ, zₙ, σ, ρ, cₚ, F)
    end
end

Nu(id::PlanarHeatsourceDynamics)::Int = 3 # vx, vz, P (m/s for speeds, W for power)
Nr(id::PlanarHeatsourceDynamics)::Int = 2 # torch position (x,z)

input_min(id::PlanarHeatsourceDynamics) = [-0.03; -0.03; 20.0] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::PlanarHeatsourceDynamics) = [0.03; 0.03; 500.0] # fourth input is std slack
input_idle(id::PlanarHeatsourceDynamics) = [0.0; 0.0; 0.0]

state_min(id::PlanarHeatsourceDynamics) = [0.0; 0.0]
state_max(id::PlanarHeatsourceDynamics) = [(id.ncols + 1) * id.l; (id.nrows + 1) * id.l]

function dynamics_function!(id::PlanarHeatsourceDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ = r[1], r[2]
    vx, vz, P = u[1], u[2], u[3]
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ

    dr[1] = vx
    dr[2] = vz
end

function input_function!(id::PlanarHeatsourceDynamics, ds::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    dT = ds
    n_rows, n_cols = id.nrows, id.ncols
    xₜ, zₜ = r[1], r[2]
    vx, vz, P = u[1], u[2], u[3]
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    ρ, cₚ = id.ρ, id.cₚ

    thread::Int = Threads.threadid()
    F = get!(id.F_cache, (Ty, thread)) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    @. F = (l^2 / (2π * σ^2)) * exp(-((xₙ - xₜ)^2 + (zₙ - zₜ)^2) / (2 * σ^2))

    # if eltype(F) == Symbolics.Num
    #     @. F += xₜ + zₜ + P
    # end

    # Forced / input dynamics
    @. dT += F * P / (ρ * l^3 * cₚ)             # Add in torch power
end