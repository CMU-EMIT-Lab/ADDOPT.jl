
function rollout!(dynamics, Nk::Int, x₀::V, X::Vector{V}, U::Vector{V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    X[1] .= x₀
    p = Progress(Nk-1)

    for k in 1:(Nk-1)
        transition!(dynamics[k], X[k+1], X[k], U[k])
        next!(p)
    end
end

function rollout!(dynamics, Nk::Int, x₀::V1, X::Vector{V2}, U::Vector{V1}) where {V1<:CuVector,V2<:Vector}
    X[1] .= V2(x₀)

    xₖ = copy(x₀)
    xₖ₊₁ = copy(x₀)
    p = Progress(Nk-1)

    for k in 1:(Nk-1)
        transition!(dynamics[k], xₖ₊₁, xₖ, U[k])

        X[k+1] .= V2(xₖ₊₁)
        xₖ .= xₖ₊₁
        next!(p)
    end
end

function rollout!(dynamics, Nk::Int, x₀::V1, X::Vector{V2}, U::Vector{V2}) where {V1<:CuVector,V2<:Vector}
    X[1] .= V2(x₀)

    xₖ = copy(x₀)
    xₖ₊₁ = copy(x₀)
    p = Progress(Nk-1)

    for k in 1:(Nk-1)
        transition!(dynamics[k], xₖ₊₁, xₖ, V1(U[k]))

        X[k+1] .= V2(xₖ₊₁)
        xₖ .= xₖ₊₁
        next!(p)
    end
end

function rollout!(process::Process, trajectory::Trajectory, x₀, U)
    p = Progress(process.Nk-1)

    for k in 1:(process.Nk-1)
        trajectory.U[k] .= U[k]
        next!(p)
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