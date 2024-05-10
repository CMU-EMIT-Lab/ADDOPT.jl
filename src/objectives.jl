using LinearAlgebra
using SparseArrays
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

struct QuadraticObjective{MatrixType<:AbstractMatrix} <: Objective
    Q::MatrixType
    b::Vector{Float64}
    R::MatrixType
    Qf::MatrixType
    x̄::Vector{Float64}
    ū::Vector{Float64}
end

function cost(o::QuadraticObjective, z, idx)
    Q, b, R, Qf, x̄, ū = o.Q, o.b, o.R, o.Qf, o.x̄, o.ū
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb[c]
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū
            cost += 0.5 * (dot(ex, Q, ex) + dot(eu, R, eu))
            cost += dot(b, ex)
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                cost += 0.5 * (dot(ex, Q, ex))
                cost += dot(b, ex)
            end
        end
    end

    xₙ = @view z[idx.x[Nc][end]]
    @. ex = xₙ - x̄
    cost += 0.5 * dot(ex, Qf, ex)
    cost += dot(b, ex)

    return cost
end

function gradient(o::QuadraticObjective, grad, z, idx)
    Q, b, R, Qf, x̄, ū = o.Q, o.b, o.R, o.Qf, o.x̄, o.ū
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb[c]
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]

            @. ex = xₖ - x̄
            @. eu = uₖ - ū

            mul!(view(grad, idx.x[c][k]), Q, ex)
            mul!(view(grad, idx.u[c][k]), R, eu)
            view(grad, idx.x[c][k]) .+= b
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                mul!(view(grad, idx.x[c][k]), Q, ex)
                view(grad, idx.x[c][k]) .+= b
            end
        end
    end

    xₙ = @view z[idx.x[Nc][end]]
    @. ex = xₙ - x̄
    mul!(view(grad, idx.x[Nc][end]), Qf, ex)
    view(grad, idx.x[Nc][end]) .+= b
end

# Hessian structure for diagonal matrices
function objective_hessian_structure(o::QuadraticObjective{Diagonal{Float64,Vector{Float64}}}, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:Nkb[c]
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])

            append!(rows, idx.u[c][k])
            append!(cols, idx.u[c][k])
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                append!(rows, idx.x[c][k])
                append!(cols, idx.x[c][k])
            end
        end
    end

    append!(rows, idx.x[Nc][end])
    append!(cols, idx.x[Nc][end])

    return collect(zip(rows, cols))
end

# Hessian evaluation for diagonal matrices
function objective_hessian_values(o::QuadraticObjective{Diagonal{Float64,Vector{Float64}}}, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    Qv = diag(o.Q)
    Rv = diag(o.R)
    Qfv = diag(o.Qf)

    i = 0
    for c in 1:Nc
        for k in 1:Nkb[c]
            H[(1+i):(length(Qv)+i)] .= Qv
            i += length(Qv)
            H[(1+i):(length(Rv)+i)] .= Rv
            i += length(Rv)
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                H[(1+i):(length(Qv)+i)] .= Qv
                i += length(Qv)
            end
        end
    end

    H[(1+i):(length(Qfv)+i)] .= Qfv
    i += length(Qfv)
end

# Hessian structure for dense matrices
function objective_hessian_structure(o::QuadraticObjective{Matrix{Float64}}, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    Qs = sparse(o.Q)
    Rs = sparse(o.R)
    Qfs = sparse(o.Qf)

    Qr, Qc, _ = findnz(Qs)
    Rr, Rc, _ = findnz(Rs)
    Qfr, Qfc, _ = findnz(Qfs)

    for c in 1:Nc
        for k in 1:Nkb[c]
            append!(rows, Qr .+ idx.x[c][k][1] .- 1)
            append!(cols, Qc .+ idx.x[c][k][1] .- 1)

            append!(rows, Rr .+ idx.u[c][k][1] .- 1)
            append!(cols, Rc .+ idx.u[c][k][1] .- 1)
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                append!(rows, Qr .+ idx.x[c][k][1] .- 1)
                append!(cols, Qc .+ idx.x[c][k][1] .- 1)
            end
        end
    end

    append!(rows, Qfr .+ idx.x[Nc][end][1] .- 1)
    append!(cols, Qfc .+ idx.x[Nc][end][1] .- 1)

    return collect(zip(rows, cols))
end

# Hessian evaluation for dense matrices
function objective_hessian_values(o::QuadraticObjective{Matrix{Float64}}, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    Qv = nonzeros(sparse(o.Q))
    Rv = nonzeros(sparse(o.R))
    Qfv = nonzeros(sparse(o.Qf))

    i = 0
    for c in 1:Nc
        for k in 1:Nkb[c]
            H[(1+i):(length(Qv)+i)] .= Qv
            i += length(Qv)
            H[(1+i):(length(Rv)+i)] .= Rv
            i += length(Rv)
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            if (c < Nc) || (k < Nkb[c] + Nkc[c])
                H[(1+i):(length(Qv)+i)] .= Qv
                i += length(Qv)
            end
        end
    end

    H[(1+i):(length(Qfv)+i)] .= Qfv
    i += length(Qfv)
end


struct TimeWeightedQuadraticObjective <: Objective
    Q::Vector{Diagonal{Float64,Vector{Float64}}}
    R::Diagonal{Float64,Vector{Float64}}
    x̄::Vector{Vector{Float64}}
    ū::Vector{Float64}
    Δt̄b::Float64
end

function cost(o::TimeWeightedQuadraticObjective, z, idx)
    Q, R, x̄, ū, Δt̄b = o.Q, o.R, o.x̄, o.ū, o.Δt̄b
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb[c]
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]
            Δt = z[idx.Δtb[c][k]]

            @. ex = xₖ - x̄[zi]
            @. eu = uₖ - ū

            cost += 0.5 * dot(ex, Q[zi], ex)
            cost += 0.5 * dot(eu, R, eu)
            cost += 0.5 * (Δt - Δt̄b)^2
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            xₖ = @view z[idx.x[c][k]]
            Δt = z[idx.Δtc[c][k]]

            @. ex = xₖ - x̄[zi]

            cost += 0.5 * dot(ex, Q[zi], ex)
            cost += 0.5 * Δt^2
        end
    end

    return cost
end

function gradient(o::TimeWeightedQuadraticObjective, grad, z, idx)
    Q, R, x̄, ū, Δt̄b = o.Q, o.R, o.x̄, o.ū, o.Δt̄b
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb[c]
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]
            Δt = z[idx.Δtb[c][k]]

            @. ex = xₖ - x̄[zi]
            @. eu = uₖ - ū

            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
            mul!(view(grad, idx.u[c][k]), R, eu)
            grad[idx.Δtb[c][k]] = Δt - Δt̄b
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            xₖ = @view z[idx.x[c][k]]
            Δt = z[idx.Δtc[c][k]]

            @. ex = xₖ - x̄[zi]

            grad[idx.Δtc[c][k]] = Δt
            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
        end
    end

end

function objective_hessian_structure(o::TimeWeightedQuadraticObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:Nkb[c]
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])

            append!(rows, idx.u[c][k])
            append!(cols, idx.u[c][k])

            append!(rows, idx.Δtb[c][k] * ones(Int, 1))
            append!(cols, idx.Δtb[c][k] * ones(Int, 1))
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])

            append!(rows, idx.Δtc[c][k] * ones(Int, 1))
            append!(cols, idx.Δtc[c][k] * ones(Int, 1))
        end
    end

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::TimeWeightedQuadraticObjective, idx, H, z)
    Q, R, x̄, ū = o.Q, o.R, o.x̄, o.ū
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    i = 0
    for c in 1:Nc
        for k in 1:Nkb[c]
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            H[(1+i):(Nx+i)] .= diag(Q[zi])
            i += Nx

            H[(1+i):(Nu+i)] .= diag(R)
            i += Nu

            H[(1+i):(1+i)] .= 1
            i += 1
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            zi = (c - 1) * (Nkb[c] + Nkc[c]) + k
            H[(1+i):(Nx+i)] .= diag(Q[zi])
            i += Nx

            H[(1+i):(1+i)] .= 1
            i += 1
        end
    end

end


struct QuadraticTrackingObjective <: Objective
    Q::Vector{Diagonal{Float64,Vector{Float64}}}
    x̄::Vector{Vector{Float64}}
end

function cost(o::QuadraticTrackingObjective, z, idx)
    Q, x̄ = o.Q, o.x̄
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    for c in 1:Nc
        for k in 1:(Nkb[c]+Nkc[c])
            zi = kc2zi(k, c, idx)
            xₖ = @view z[idx.x[c][k]]
            @. ex = xₖ - x̄[zi]

            cost += 0.5 * dot(ex, Q[zi], ex)
        end
    end

    return cost
end

function gradient(o::QuadraticTrackingObjective, grad, z, idx)
    Q, x̄ = o.Q, o.x̄
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    for c in 1:Nc
        for k in 1:(Nkb[c]+Nkc[c])
            zi = kc2zi(k, c, idx)
            xₖ = @view z[idx.x[c][k]]
            @. ex = xₖ - x̄[zi]

            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
        end
    end

end

function objective_hessian_structure(o::QuadraticTrackingObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:(Nkb[c]+Nkc[c])
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])
        end
    end

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::QuadraticTrackingObjective, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nx, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Q = o.Q

    i = 0
    for c in 1:Nc
        for k in 1:(Nkb[c]+Nkc[c])
            zi = kc2zi(k, c, idx)
            H[(1+i):(Nx+i)] .= diag(Q[zi])
            i += Nx
        end
    end
end