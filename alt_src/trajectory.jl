struct Process{T<:AbstractFloat,V<:AbstractVector{T},D<:Dynamics{T}}
    Nk::Int                       # Number of timesteps
    dynamics::Vector{D} # Dynamics for each timestep, Nk long

    function Process(Nk::Int, dynamics::Vector{D}, V) where {T<:AbstractFloat,D<:Dynamics{T}}
        @assert Nk == length(dynamics) "Number of dynamics doesn't match number of timesteps"

        new{T,V,D}(Nk, dynamics)
    end
end

struct Trajectory{T<:AbstractFloat,V<:AbstractVector{T}}
    Nk::Int      # Number of timesteps
    X::Vector{V} # State trajectory, Nk long
    U::Vector{V} # Input trajectory, Nk long (last element 0)

    # Initialize trajectory for process
    function Trajectory(process::Process{T,V,D}) where {T<:AbstractFloat,V<:AbstractVector{T},D<:Dynamics{T}}
        Nk = process.Nk
        dynamics = process.dynamics

        X::Vector{V} = [V(undef, nx(dynamics[k])) for k in 1:Nk]
        U::Vector{V} = [V(undef, nu(dynamics[k])) for k in 1:Nk]

        U[end] .= 0

        new{T,V}(Nk, X, U)
    end

    # Initialize trajectory based on the dimensions of an existing one
    function Trajectory(trajectory::Trajectory{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
        Nk = trajectory.Nk
        X, U = trajectory.X, trajectory.U

        X::Vector{V} = [V(undef, length(x)) for x in X]
        U::Vector{V} = [V(undef, length(u)) for u in U]
        U[end] .= 0

        new{T,V}(Nk, X, U)
    end
end

function copy!(z::Trajectory{T,V}, z̄::Trajectory{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    Nk = z.Nk

    for k in 1:Nk
        z.X[k] .= z̄.X[k]
        z.U[k] .= z̄.U[k]
    end
end

struct ConstraintTrajectory{T<:AbstractFloat,V<:AbstractVector{T}}
    Nk::Int                     # Number of timesteps
    c::Vector{V}                # Constraint evaluations, Nk long
    λ::Vector{V}                # Dual variables, Nk long
    Iμ::Vector{Diagonal{T,V}}   # Diagonal complementarity matrix

    function ConstraintTrajectory(process::Process{T,V,D}, constraints) where {T<:AbstractFloat,V<:AbstractVector{T},D<:Dynamics{T}}
        Nk = process.Nk

        c::Vector{V} = [V(undef, nc(constraints[k])) for k in 1:Nk]
        λ::Vector{V} = [V(undef, nc(constraints[k])) for k in 1:Nk]
        Iμ::Vector{Diagonal{T,V}} = [Diagonal(V(undef, nc(constraints[k]))) for k in 1:Nk]

        new{T,V}(Nk, c, λ, Iμ)
    end
end


function copy!(v::ConstraintTrajectory{T,V}, v̄::ConstraintTrajectory{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    Nk = v.Nk

    for k in 1:Nk
        v.c[k] .= v̄.c[k]
        v.λ[k] .= v̄.λ[k]
        v.Iμ[k].diag .= v̄.Iμ[k].diag
    end
end

function eval_lagrangian_cost(v::ConstraintTrajectory{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    c, Iμ, λ = v.c, v.Iμ, v.λ
    Nk = v.Nk

    cost::T = zero(T)
    for k in 1:Nk
        ck::V = c[k]
        λk::V = λ[k]
        Iμk::Diagonal{T,V} = Iμ[k]

        cost += dot(λk, ck)
        cost += 0.5 * mapreduce((c, d) -> c^2 * d, +, ck, Iμk.diag)
    end

    return cost
end