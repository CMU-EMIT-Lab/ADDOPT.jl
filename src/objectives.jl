using LinearAlgebra
using ForwardDiff

abstract type Objective end

function cost(o::Objective, z, idx)
    return 0.0
end

function gradient(o::Objective, grad, z, idx)
    ForwardDiff.gradient!(grad, (Z) -> cost(o, Z, idx), z)
end

function hessian(o::Objective, hess, z, idx)
    ForwardDiff.hessian!(hess, (Z) -> cost(o, Z, idx), z)
end

struct QuadraticObjective <: Objective
    Q::Diagonal{Float64,Vector{Float64}}
    R::Diagonal{Float64,Vector{Float64}}
    Qf::Diagonal{Float64,Vector{Float64}}
    x̄::Vector{Float64}
    ū::Vector{Float64}
    tw::Float64
end

function cost(o::QuadraticObjective, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū
            # if !isnothing(idx.Δt)
            #     cost += o.tw * z[idx.Δt[c][k]]
            # end
            cost += 0.5 * (dot(ex, Q, ex) + dot(eu, R, eu))# * (isnothing(idx.Δt) ? 1.0 : z[idx.Δt[c][k]])
        end
    end

    xₙ = @view z[idx.x[Nc][Nk]]
    @. ex = xₙ - x̄
    cost += 0.5 * dot(ex, Qf, ex)


    return cost
end

function gradient(o::QuadraticObjective, grad, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū

            mul!(view(grad, idx.x[c][k]), Q, ex)
            mul!(view(grad, idx.u[c][k]), R, eu)
            # if !isnothing(idx.Δt)
            #     grad[idx.Δt[c][k]] = o.tw + 0.5 * (dot(ex, Q, ex) + dot(eu, R, eu))
            #     view(grad, idx.x[c][k]) .*= z[idx.Δt[c][k]]
            #     view(grad, idx.u[c][k]) .*= z[idx.Δt[c][k]]
            # end
        end
    end

    xₙ = @view z[idx.x[Nc][Nk]]
    @. ex = xₙ - x̄
    mul!(view(grad, idx.x[Nc][Nk]), Qf, ex)

end

struct MinTimeObjective <: Objective
    tw::Float64
end

function cost(o::MinTimeObjective, z, idx)
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu

    cost = 0.0
    for c in 1:Nc

        for k in 1:Nk
            cost += o.tw * z[idx.Δt[c][k]]
        end
    end

    return cost
end

function gradient(o::MinTimeObjective, grad, z, idx)
    Nx, Nk, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nc, idx.Nu
    grad[:] .= 0

    for c in 1:Nc

        for k in 1:Nk
            grad[idx.Δt[c][k]] = o.tw
        end
    end

end