
function rollout!(dynamics, Nk::Int, x₀::V, X::Vector{V}, U::Vector{V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    X[1] .= x₀

    for k in 1:(Nk-1)
        transition!(dynamics[k], X[k+1], X[k], U[k])
    end
end

function rollout!(process::Process, trajectory::Trajectory, x₀, U)
    for k in 1:(process.Nk-1)
        trajectory.U[k] .= U[k]
    end

    rollout!(process, trajectory, x₀)
end

function rollout!(process::Process, trajectory::Trajectory, x₀)
    dynamics = process.dynamics
    Nk = process.Nk
    X = trajectory.X
    U = trajectory.U

    rollout!(dynamics, Nk::Int, x₀, X, U)
end