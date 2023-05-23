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
end

function hessian_structure(o::QuadraticObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:Nkb
            append!(rows,  idx.x[c][k])
            append!(cols,  idx.x[c][k])

            append!(rows,  idx.u[c][k])
            append!(cols,  idx.u[c][k])
        end

        for k in (Nkb+1):(Nkb+Nkc)
            if (c < Nc) || (k < Nkb+Nkc)
                append!(rows,  idx.x[c][k])
                append!(cols,  idx.x[c][k])
            end
        end
    end

    append!(rows, idx.x[Nc][Nkb+Nkc])
    append!(cols, idx.x[Nc][Nkb+Nkc])

    return collect(zip(rows, cols))
end

function hessian_values(o::QuadraticObjective, idx, H)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    Qv = diag(o.Q)
    Rv = diag(o.R)
    Qfv = diag(o.Qf)

    i = 0
    for c in 1:Nc
        for k in 1:Nkb
            H[(1+i):(length(Qv)+i)] .= Qv
            i += length(Qv)
            H[(1+i):(length(Rv)+i)] .= Rv
            i += length(Rv)
        end

        for k in (Nkb+1):(Nkb+Nkc)
            if (c < Nc) || (k < Nkb+Nkc)
                H[(1+i):(length(Qv)+i)] .= Qv
                i += length(Qv)
            end
        end
    end

    H[(1+i):(length(Qfv)+i)] .= Qfv
    i += length(Qfv)
end

function cost(o::QuadraticObjective, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū
            # if !isnothing(idx.Δt)
            #     cost += o.tw * z[idx.Δt[c][k]]
            # end
            cost += 0.5 * (dot(ex, Q, ex) + dot(eu, R, eu))# * (isnothing(idx.Δt) ? 1.0 : z[idx.Δt[c][k]])
        end

        for k in (Nkb+1):(Nkb+Nkc)
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb+Nkc)
                cost += 0.5 * (dot(ex, Q, ex))# * (isnothing(idx.Δt) ? 1.0 : z[idx.Δt[c][k]])
            end
        end
    end

    xₙ = @view z[idx.x[Nc][Nkb+Nkc]]
    @. ex = xₙ - x̄
    cost += 0.5 * dot(ex, Qf, ex)


    return cost
end

function gradient(o::QuadraticObjective, grad, z, idx)
    Q, R, Qf, x̄, ū = o.Q, o.R, o.Qf, o.x̄, o.ū
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb
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

        for k in (Nkb+1):(Nkb+Nkc)
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb+Nkc)
                mul!(view(grad, idx.x[c][k]), Q, ex)
            end
        end
    end

    xₙ = @view z[idx.x[Nc][Nkb+Nkc]]
    @. ex = xₙ - x̄
    mul!(view(grad, idx.x[Nc][Nkb+Nkc]), Qf, ex)

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