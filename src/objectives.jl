using LinearAlgebra

abstract type Objective end

function cost(o::Objective, z, idx)
    return 0.0
end

struct QuadraticObjective <: Objective
    Q
    R
    Qf
    x̄
    ū
    tw
end

function cost(o::QuadraticObjective, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc
        cost += o.tw * z[idx.Δt[c]]
        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū
            cost += 0.5 * (ex' * Q * ex + eu' * R * eu)
        end
    end

    xₙ = @view z[idx.x[Nc][Nk]]
    @. ex = xₙ - x̄
    cost += 0.5 * (ex' * Qf * ex)


    return cost
end

function gradient(o::QuadraticObjective, grad, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc
        grad[idx.Δt[c]] = o.tw
        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū
            # @show size(grad[idx.x[c][k]])
            # @show size(Q*ex)
            grad[idx.x[c][k]] .= Q * ex
            grad[idx.u[c][k]] .= R * eu
        end
    end

    xₙ = @view z[idx.x[Nc][Nk]]
    @. ex = xₙ - x̄
    grad[idx.x[Nc][Nk]] .= Qf * ex

end