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
    R::Float64
    Qf::Diagonal{Float64,Vector{Float64}}
    xg::Vector{Float64}
end

function cost(o::CoolingTrackingObjective, z, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xg = o.Qf, o.xg
    R = o.R
    Nvox = Nx ÷ 2

    ex = zeros(eltype(z), Nvox)
    eR = zeros(eltype(z), Nvox)

    cost = 0.0
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            if c == Nc && k == Nkb + Nkc
                continue
            end
            zi = kc2zi(k, c, idx)

            xₖ = @view z[(idx.x[c][k])[(Nvox+1):2Nvox]]
            Eₖ = @view z[(idx.x[c][k])[1:Nvox]]
            Eₖ₊₁ = k == Nkb + Nkc ? (@view z[(idx.x[c+1][1])[1:Nvox]]) : (@view z[(idx.x[c][k+1])[1:Nvox]])
            Δt = isnothing(idx.Δtb) ? 0.04 : (k > Nkb ? (z[idx.Δtc[c][k]]) : (z[idx.Δtb[c][k]]))
            @. ex = xₖ - x̄[zi]
            @. eR = Eₖ - Eₖ₊₁ - R * Δt

            cost += 0.5 * (dot(ex, Qxs[zi], ex) * Δt + dot(eR, QRs[zi], eR))
        end
    end

    xf = @view z[idx.x[Nc][Nkb+Nkc]]
    ex = xf .- xg
    cost += 0.5 * (dot(ex, Qf, ex))

    return cost
end

function gradient(o::CoolingTrackingObjective, grad, z, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xg = o.Qf, o.xg
    R = o.R
    Nvox = Nx ÷ 2

    ex = zeros(eltype(z), Nvox)
    eR = zeros(eltype(z), Nvox)

    grad .= 0.0
    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            if c == Nc && k == Nkb + Nkc
                continue
            end
            zi = kc2zi(k, c, idx)

            Enidx = k == Nkb + Nkc ? ((idx.x[c+1][1])[1:Nvox]) : ((idx.x[c][k+1])[1:Nvox])
            Δtidx = isnothing(idx.Δtb) ? 0.04 : (k > Nkb ? (idx.Δtc[c][k]) : (idx.Δtb[c][k]))

            xₖ = @view z[(idx.x[c][k])[(Nvox+1):2Nvox]]
            Eₖ = @view z[(idx.x[c][k])[1:Nvox]]
            Eₖ₊₁ = @view z[Enidx]
            Δt = isnothing(idx.Δtb) ? 0.04 : z[Δtidx]

            @. ex = xₖ - x̄[zi]
            @. eR = Eₖ - Eₖ₊₁ - R * Δt

            grad[(idx.x[c][k])[(Nvox+1):2Nvox]] .+= Δt * Qxs[zi] * ex
            grad[(idx.x[c][k])[1:Nvox]] .+= QRs[zi] * eR
            grad[Enidx] .+= -QRs[zi] * eR
            if !isnothing(idx.Δtb)
                grad[Δtidx] += 0.5 * dot(ex, Qxs[zi], ex) .- R * ones(Nvox)' * QRs[zi] * eR
            end
        end
    end

    xf = @view z[idx.x[Nc][Nkb+Nkc]]
    ex = xf .- xg
    grad[idx.x[Nc][Nkb+Nkc]] .+= Qf * ex
end

function objective_hessian_structure(o::CoolingTrackingObjective, idx)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    R = o.R
    Nvox = Nx ÷ 2

    rows = []
    cols = []

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            if c == Nc && k == Nkb + Nkc
                continue
            end

            Enidx = k == Nkb + Nkc ? ((idx.x[c+1][1])[1:Nvox]) : ((idx.x[c][k+1])[1:Nvox])
            Δtidx = Δt = isnothing(idx.Δtb) ? 0.04 : (k > Nkb ? (idx.Δtc[c][k]) : (idx.Δtb[c][k]))

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

    append!(rows, idx.x[Nc][Nkb+Nkc])
    append!(cols, idx.x[Nc][Nkb+Nkc])

    return collect(zip(rows, cols))
end

function objective_hessian_values(o::CoolingTrackingObjective, idx, H, z)
    Nx, Nkb, Nkc, Nc, Nu = idx.Nstates, idx.Nkb, idx.Nkc, idx.Nc, idx.Nu
    Qxs, QRs, x̄ = o.Qxs, o.QRs, o.x̄
    Qf, xg = o.Qf, o.xg
    R = o.R
    Nvox = Nx ÷ 2

    i = 0
    H .= 0.0

    for c in 1:Nc
        for k in 1:(Nkb+Nkc)
            if c == Nc && k == Nkb + Nkc
                continue
            end
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

    H[(1+i):(2Nvox+i)] .+= diag(Qf)
    i += 2Nvox
end