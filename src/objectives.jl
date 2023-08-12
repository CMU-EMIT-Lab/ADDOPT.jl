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
            cost += 0.5 * (dot(ex, Q, ex) + dot(eu, R, eu))
        end

        for k in (Nkb+1):(Nkb+Nkc)
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb + Nkc)
                cost += 0.5 * (dot(ex, Q, ex))
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
        end

        for k in (Nkb+1):(Nkb+Nkc)
            xₖ = @view z[idx.x[c][k]]

            @. ex = xₖ - x̄

            if (c < Nc) || (k < Nkb + Nkc)
                mul!(view(grad, idx.x[c][k]), Q, ex)
            end
        end
    end

    xₙ = @view z[idx.x[Nc][Nkb+Nkc]]
    @. ex = xₙ - x̄
    mul!(view(grad, idx.x[Nc][Nkb+Nkc]), Qf, ex)

end

function objective_hessian_structure(o::QuadraticObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:Nkb
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])

            append!(rows, idx.u[c][k])
            append!(cols, idx.u[c][k])
        end

        for k in (Nkb+1):(Nkb+Nkc)
            if (c < Nc) || (k < Nkb + Nkc)
                append!(rows, idx.x[c][k])
                append!(cols, idx.x[c][k])
            end
        end
    end

    append!(rows, idx.x[Nc][Nkb+Nkc])
    append!(cols, idx.x[Nc][Nkb+Nkc])

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::QuadraticObjective, idx, H, z)
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
            if (c < Nc) || (k < Nkb + Nkc)
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
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb
            zi = (c - 1) * (Nkb + Nkc) + k
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]
            Δt = z[idx.Δtb[c][k]]

            @. ex = xₖ - x̄[zi]
            @. eu = uₖ - ū

            cost += 0.5 * dot(ex, Q[zi], ex)
            cost += 0.5 * dot(eu, R, eu)
            cost += 0.5 * (Δt-Δt̄b)^2
        end

        for k in (Nkb+1):(Nkb+Nkc)
            zi = (c - 1) * (Nkb + Nkc) + k
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
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    eu = zeros(eltype(z), Nu)
    for c in 1:Nc

        for k in 1:Nkb
            zi = (c - 1) * (Nkb + Nkc) + k
            xₖ = @view z[idx.x[c][k]]
            uₖ = @view z[idx.u[c][k]]
            Δt = z[idx.Δtb[c][k]]

            @. ex = xₖ - x̄[zi]
            @. eu = uₖ - ū

            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
            mul!(view(grad, idx.u[c][k]), R, eu)
            grad[idx.Δtb[c][k]] = Δt-Δt̄b
        end

        for k in (Nkb+1):(Nkb+Nkc)
            zi = (c - 1) * (Nkb + Nkc) + k
            xₖ = @view z[idx.x[c][k]]
            Δt = z[idx.Δtc[c][k]]

            @. ex = xₖ - x̄[zi]

            grad[idx.Δtc[c][k]] = Δt
            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
        end
    end

end

function objective_hessian_structure(o::TimeWeightedQuadraticObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:Nkb
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])

            append!(rows, idx.u[c][k])
            append!(cols, idx.u[c][k])

            append!(rows, idx.Δtb[c][k] * ones(Int, 1))
            append!(cols, idx.Δtb[c][k] * ones(Int, 1))
        end

        for k in (Nkb+1):(Nkb+Nkc)
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
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    i = 0
    for c in 1:Nc
        for k in 1:Nkb
            zi = (c - 1) * (Nkb + Nkc) + k
            H[(1+i):(Nx+i)] .= diag(Q[zi])
            i += Nx

            H[(1+i):(Nu+i)] .= diag(R)
            i += Nu

            H[(1+i):(1+i)] .= 1
            i += 1
        end

        for k in (Nkb+1):(Nkb+Nkc)
            zi = (c - 1) * (Nkb + Nkc) + k
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
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    ex = zeros(eltype(z), Nx)
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
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
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    ex = zeros(eltype(z), Nx)
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            zi = kc2zi(k, c, idx)
            xₖ = @view z[idx.x[c][k]]
            @. ex = xₖ - x̄[zi]

            mul!(view(grad, idx.x[c][k]), Q[zi], ex)
        end
    end

end

function objective_hessian_structure(o::QuadraticTrackingObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            append!(rows, idx.x[c][k])
            append!(cols, idx.x[c][k])
        end
    end

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::QuadraticTrackingObjective, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Q = o.Q

    i = 0
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            zi = kc2zi(k, c, idx)
            H[(1+i):(Nx+i)] .= diag(Q[zi])
            i += Nx
        end
    end
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


struct CoolingTrackingObjective <: Objective
    Qxs::Vector{Diagonal{Float64,Vector{Float64}}}
    QRs::Vector{Diagonal{Float64,Vector{Float64}}}
    x̄::Vector{Vector{Float64}}
    fill_ref::Vector{Vector{Int}}
    R::Float64
    Emax::Float64
    E∞::Float64
    Qf::Diagonal{Float64,Vector{Float64}}
    xf::Vector{Float64}
end

function Ē(o::CoolingTrackingObjective, t)
    if t > 0.0
        return (o.Emax - o.E∞) * exp(-t / o.R) + o.E∞
    else
        return 0.0#o.E∞
    end
end

function ∂Ē(o::CoolingTrackingObjective, t)
    if t > 0.0
        return (o.Emax - o.E∞) / o.R * exp(-t / o.R)
    else
        return 0.0
    end
end


function cost(o::CoolingTrackingObjective, z, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xf = o.Qf, o.xf
    R = o.R
    Nvox = Nx ÷ 2

    ex = zeros(eltype(z), Nvox)
    exN = zeros(eltype(z), Nx)
    eR = zeros(eltype(z), Nvox)
    t = zeros(eltype(z), Nvox)
    final_fill = o.fill_ref[end]
    Er(t) = Ē(o, t)

    cost = 0.0
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            zi = kc2zi(k, c, idx)

            xₖ = @view z[(idx.x[c][k])[(Nvox+1):2Nvox]]
            Eₖ = @view z[(idx.x[c][k])[1:Nvox]]
            Δt = k > Nkb ? (z[idx.Δtc[c][k]]) : (z[idx.Δtb[c][k]])
            t[k.==final_fill] .+= Δt

            @. ex = xₖ - x̄[zi]
            @. eR = Eₖ - Er(t)

            cost += 0.5 * (dot(ex, Qxs[zi], ex) + dot(eR, QRs[zi], eR))

            if k > Nkb
                xN = @view z[(idx.x[c][k])]
                @. exN = xN - xf
                cost += 1e-2 * 0.5 * dot(exN, Qf, exN)
            end
        end
    end

    xN = @view z[idx.x[Nc][Nkb+Nkc]]
    @. exN = xN - xf
    cost += 0.5 * dot(exN, Qf, exN)

    return cost
end

function gradient(o::CoolingTrackingObjective, grad, z, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xf = o.Qf, o.xf
    R = o.R
    Nvox = Nx ÷ 2

    ex = zeros(eltype(z), Nvox)
    exN = zeros(eltype(z), Nx)
    eR = zeros(eltype(z), Nvox)
    t = zeros(eltype(z), Nvox)
    mask = zeros(eltype(z), Nvox)
    final_fill = o.fill_ref[end]
    Er(t) = Ē(o, t)
    ∂Er(t) = ∂Ē(o, t)

    grad .= 0.0
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            zi = kc2zi(k, c, idx)

            Δtidx = k > Nkb ? (idx.Δtc[c][k]) : (idx.Δtb[c][k])

            xₖ = @view z[(idx.x[c][k])[(Nvox+1):2Nvox]]
            Eₖ = @view z[(idx.x[c][k])[1:Nvox]]
            Δt = z[Δtidx]
            t[k.==final_fill] .+= Δt

            @. ex = xₖ - x̄[zi]
            @. eR = Eₖ - Er(t)

            grad[(idx.x[c][k])[(Nvox+1):2Nvox]] .+= Qxs[zi] * ex
            grad[(idx.x[c][k])[1:Nvox]] .+= QRs[zi] * eR

            for j in 1:k
                mask .= 0
                mask[j.==final_fill] .= 1
                Δtjidx = j > Nkb ? (idx.Δtc[c][j]) : (idx.Δtb[c][j])
                grad[Δtjidx] += dot(∂Er.(t), QRs[zi], eR .* mask)
            end

            if k > Nkb
                xN = @view z[(idx.x[c][k])]
                @. exN = xN - xf
                grad[idx.x[c][k]] .+= 1e-2 * Qf * exN
            end
        end
    end

    xN = @view z[idx.x[Nc][Nkb+Nkc]]
    @. exN = xN - xf
    grad[idx.x[Nc][Nkb+Nkc]] .+= Qf * exN
end

function objective_hessian_structure(o::CoolingTrackingObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xf = o.Qf, o.xf
    R = o.R
    Nvox = Nx ÷ 2

    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)

            Enidx = k == Nkb + Nkc ? ((idx.x[c+1][1])[1:Nvox]) : ((idx.x[c][k+1])[1:Nvox])
            Δtidx = k > Nkb ? (idx.Δtc[c][k]) : (idx.Δtb[c][k])

            xₖ = (idx.x[c][k])[(Nvox+1):2Nvox]
            Eₖ = (idx.x[c][k])[1:Nvox]
            Eₖ₊₁ = Enidx
            Δt = Δtidx

            append!(rows, xₖ)
            append!(cols, xₖ)

            append!(rows, Eₖ)
            append!(cols, Eₖ)

            append!(rows, Eₖ₊₁)
            append!(cols, Eₖ₊₁)

            if !isnothing(idx.Δtb)
                append!(rows, Δt)
                append!(cols, Δt)
            end

            append!(rows, Eₖ)
            append!(cols, Eₖ₊₁)

            if !isnothing(idx.Δtb)
                append!(rows, xₖ)
                append!(cols, Δt * ones(Int, Nvox))

                append!(rows, Eₖ)
                append!(cols, Δt * ones(Int, Nvox))

                append!(rows, Eₖ₊₁)
                append!(cols, Δt * ones(Int, Nvox))
            end
        end
    end


    return collect(zip(rows, cols))
end

function objective_hessian_values(o::CoolingTrackingObjective, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xf = o.Qf, o.xf
    R = o.R
    Nvox = Nx ÷ 2

    i = 0
    H .= 0.0

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            zi = kc2zi(k, c, idx)

            Δtidx = isnothing(idx.Δtb) ? 0.04 : (k > Nkb ? (idx.Δtc[c][k]) : (idx.Δtb[c][k]))

            xₖ = @view z[(idx.x[c][k])[(Nvox+1):2Nvox]]
            Δt = isnothing(idx.Δtb) ? 0.04 : z[Δtidx]

            H[(1+i):(Nvox+i)] .+= diag(Qxs[zi]) * Δt
            i += Nvox

            H[(1+i):(Nvox+i)] .+= diag(QRs[zi])
            i += Nvox

            H[(1+i):(Nvox+i)] .+= diag(QRs[zi])
            i += Nvox

            if !isnothing(idx.Δtb)
                H[(1+i):(1+i)] .+= tr(Qxs[zi]) * R^2
                i += 1
            end

            H[(1+i):(Nvox+i)] .+= -diag(QRs[zi])
            i += Nvox

            if !isnothing(idx.Δtb)
                H[(1+i):(Nvox+i)] .+= Qxs[zi] * (xₖ - x̄[zi])
                i += Nvox

                H[(1+i):(Nvox+i)] .+= -diag(QRs[zi]) * R
                i += Nvox

                H[(1+i):(Nvox+i)] .+= diag(QRs[zi]) * R
                i += Nvox
            end
        end
    end

end


struct QuadraticCubicObjective <: Objective
    Nvox::Int
    C::Vector{Float64}
    Q::Diagonal{Float64,Vector{Float64}}
    T̄::Vector{Float64}
    ȳ::Vector{Float64}
end

function cost(o::QuadraticCubicObjective, z, idx)
    Nvox, C, Q, T̄, ȳ = o.Nvox, o.C, o.Q, o.T̄, o.ȳ
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    cost = 0.0
    eT = zeros(eltype(z), Nvox)
    ey = zeros(eltype(z), Nvox)
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            Tₖ = @view z[idx.x[c][k][1:Nvox]]
            yₖ = @view z[idx.x[c][k][(Nvox+1):2Nvox]]

            @. eT = Tₖ - T̄
            clamp!(eT, 0.0, Inf)
            eT .^= 3

            @. ey = yₖ - ȳ
            cost += dot(C, eT) + 0.5 * dot(ey, Q, ey)
        end
    end

    return cost
end

function gradient(o::QuadraticCubicObjective, grad, z, idx)
    Nvox, C, Q, T̄, ȳ = o.Nvox, o.C, o.Q, o.T̄, o.ȳ
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    grad[:] .= 0

    eT = zeros(eltype(z), Nvox)
    ey = zeros(eltype(z), Nvox)
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            Tₖ = @view z[idx.x[c][k][1:Nvox]]
            yₖ = @view z[idx.x[c][k][(Nvox+1):2Nvox]]

            @. eT = Tₖ - T̄
            clamp!(eT, 0.0, Inf)
            eT .^= 2

            @. ey = yₖ - ȳ

            grad[idx.x[c][k][1:Nvox]] .= 3 .* C .* eT
            mul!(view(grad, idx.x[c][k][(Nvox+1):2Nvox]), Q, ey)
        end
    end
end

function objective_hessian_structure(o::QuadraticCubicObjective, idx)
    Nvox, C, Q, T̄, ȳ = o.Nvox, o.C, o.Q, o.T̄, o.ȳ
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            append!(rows, idx.x[c][k][1:2Nvox])
            append!(cols, idx.x[c][k][1:2Nvox])
        end
    end

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::QuadraticCubicObjective, idx, H, z)
    Nvox, C, Q, T̄, ȳ = o.Nvox, o.C, o.Q, o.T̄, o.ȳ
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu

    Qv = diag(Q)
    i = 0
    eT = zeros(eltype(z), Nvox)
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            Tₖ = @view z[idx.x[c][k][1:Nvox]]
            @. eT = Tₖ - T̄
            clamp!(eT, 0.0, Inf)

            H[(1+i):(Nvox+i)] .= 6 .* C .* eT
            i += Nvox

            H[(1+i):(Nvox+i)] .= Qv
            i += Nvox
        end
    end
end