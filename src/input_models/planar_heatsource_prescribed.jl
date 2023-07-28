using StatsFuns

struct PlanarHeatsourcePrescribedMotionDynamics <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    σ::Float64

    ρ::Float64
    cₚ::Float64

    F_cache::Dict{Tuple{DataType, Int},Any}

    xmin::Float64
    xmax::Float64
    zmin::Float64
    zmax::Float64
    tmin::Float64
    tmax::Float64

    function PlanarHeatsourcePrescribedMotionDynamics(nrows, ncols, l, xₙ, zₙ, ρ, cₚ, σ, xmin, xmax, zmin, zmax, tmin, tmax)
        F = Dict{Tuple{DataType, Int},Any}()

        return new(nrows, ncols, l, xₙ, zₙ, σ, ρ, cₚ, F, xmin, xmax, zmin, zmax, tmin, tmax)
    end
end

Nu(id::PlanarHeatsourcePrescribedMotionDynamics)::Int = 1 # P (W)
Nr(id::PlanarHeatsourcePrescribedMotionDynamics)::Int = 0 # torch position (x,z)

input_min(id::PlanarHeatsourcePrescribedMotionDynamics) = [20.0] # vx, vz, trim, WFS (m/s for speeds) # second gausshess constrained vx to 10 mm/s
input_max(id::PlanarHeatsourcePrescribedMotionDynamics) = [500.0]
input_idle(id::PlanarHeatsourcePrescribedMotionDynamics) = [0.0]

state_min(id::PlanarHeatsourcePrescribedMotionDynamics) = []
state_max(id::PlanarHeatsourcePrescribedMotionDynamics) = []

function dynamics_function!(id::PlanarHeatsourcePrescribedMotionDynamics, dr::AbstractVector{Ty}, s, r, u, t, zi) where {Ty}

end

function input_function!(id::PlanarHeatsourcePrescribedMotionDynamics, ds::AbstractVector{Ty}, r, u, t, zi) where {Ty}
    dT = ds
    n_rows, n_cols = id.nrows, id.ncols
    P = u[1]
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    ρ, cₚ = id.ρ, id.cₚ
    xmin, xmax, zmin, zmax, tmin, tmax = id.xmin, id.xmax, id.zmin, id.zmax, id.tmin, id.tmax
    xₜ = clamp((xmax - xmin) / (tmax - tmin) * (t - tmin) + xmin, xmin, xmax)
    zₜ = clamp((zmax - zmin) / (tmax - tmin) * (t - tmin) + zmin, zmin, zmax)

    thread::Int = Threads.threadid()
    F = get!(id.F_cache, (Ty, thread)) do
        zeros(Ty, n_rows * n_cols)
    end::Vector{Ty}

    @. F = (l^2 / (2π * σ^2)) * exp(-((xₙ - xₜ)^2 + (zₙ - zₜ)^2) / (2σ^2)) # Gaussian about heat source location

    # if eltype(F) == Symbolics.Num
    #     F .= 1#@. F += xₜ + zₜ + P
    # end
    #else
    # #     F ./= sum(F)
    # end 

    # Forced / input dynamics
    @. dT += F * P / (ρ * l^3 * cₚ)             # Add in torch power
end