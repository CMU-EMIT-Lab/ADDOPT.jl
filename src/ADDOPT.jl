module ADDOPT

using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
using Symbolics, SparseArrays, SparseDiffTools
using SparseArrays: findnz
const MOI = MathOptInterface

abstract type Dynamics end

abstract type InputDynamics <: Dynamics end
abstract type TransferDynamics <: Dynamics end
abstract type PropertyDynamics <: Dynamics end

Nu(id::InputDynamics) = 0
Nr(id::InputDynamics) = 0
Nα(pd::PropertyDynamics) = 0
Ns(td::TransferDynamics) = 0

include("processes.jl")
include("objectives.jl")

struct AdditiveProblem <: MOI.AbstractNLPEvaluator
    process::Process

    objective::Objective
    # objective_gradient!::Union{Function,Nothing}
    # hessian::Union{Function,Matrix,Nothing}

    # Nx::Int
    # Ny::Int
    # l::Real

    Δt::Union{Real,Nothing}
    Nkb::Int # Number of knots per build cycle
    # Nkc::Int # Number of knots per cooling cycle
    Nc::Int  # Number of cycles

    eval_constraint!::Function
    constraint_jacobian_sparsity

    idx
    sparsity_cache
    colloc!
    jump_constraint!

    x₀
    x̄

    function AdditiveProblem(process, objective, #Nx, Ny, l, 
        Nkb, Nc, x₀; x̄=nothing, Δt=nothing)
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        free_time = isnothing(Δt)
        idx = generate_z_indices(Nkb, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), free_time=free_time)

        f!(ẋ, x, u) = combined_dynamics!(ẋ, x, u, process)
        fd!(xₖ₊₁, xₖ, Δt) = combined_jump!(xₖ₊₁, xₖ, Δt, process)

        colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = collocation_constraint!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, f!, Δt)
        jump_constraint!(r, xₖ, xₖ₊₁, Δt) = j_constraint!(r, xₖ, xₖ₊₁, fd!, Δt)
        eval_constraint!(c, z) = constraints!(c, z, idx, colloc!, jump_constraint!, x₀; xf=x̄, Δt=Δt)

        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, colloc!, jump_constraint!, Δt=Δt)

        new(process, objective, Δt, Nkb, Nc,
            eval_constraint!, con_jacobian_sparsity,
            idx, sparsity_cache, colloc!, jump_constraint!, x₀, x̄)#Nx, Ny, l,
    end
end

function generate_z_indices(Nkb, Nc, Nu, Nr, Ns, Nα; free_time=false)
    NΔt = free_time ? 1 : 0
    Nstates = (Ns + Nα)#Nx * Ny * 
    Nknotvals = Nstates + Nr + Nu
    Npercycle = (NΔt + Nknotvals * Nkb + 1)
    Nz = Npercycle * Nc #+ Nstates # Last term is for state after final cooling

    s = [[((NΔt+1):(Ns+NΔt)) .+ (Nknotvals * (k - 1) + Npercycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc] # z[s[c][k]] gives s_(c,k), vector
    α = [[((Ns+NΔt+1):(Nstates+NΔt)) .+ (Nknotvals * (k - 1) + Npercycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc] # z[α[c][k]] gives α_(c,k), vector
    r = [[((Nstates+NΔt+1):(Nstates+NΔt+Nr)) .+ (Nknotvals * (k - 1) + Npercycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc] # z[r[c][k]] gives r_(c,k), vector
    u = [[((Nstates+NΔt+Nr+1):(NΔt+Nknotvals)) .+ (Nknotvals * (k - 1) + Npercycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc] # z[u[c][k]] gives u_(c,k), vector
    x = [[((NΔt+1):(Nstates+NΔt+Nr)) .+ (Nknotvals * (k - 1) + Npercycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc]
    Δt = free_time ? [1 + Npercycle * (c - 1) for c in 1:Nc] : nothing  # z[Δt[c]] gives Δt_c, scalar
    tc = [Npercycle * c for c in 1:Nc]                                  # z[tc[c]] gives tc_c, scalar

    Nconstr = (Nstates + Nr) * (Nkb - 1) * Nc + (Nstates + Nr) * 2

    return (Nz=Nz, Nstates=(Nstates + Nr), s=s, α=α, r=r, u=u, x=x, Δt=Δt, tc=tc, Nconstr=Nconstr, Nkb=Nkb, Nc=Nc, Nu=Nu)
end

function combined_dynamics!(f, x, u, process::Process)
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics
    # Ns, Nα(pd) = Ns(td)(td), Nα(pd)(pd)
    # Nr, Nu = Nr(id), Nu(id)

    s = @view x[1:Ns(td)]
    α = @view x[(Ns(td)+1):(Ns(td)+Nα(pd))]
    r = @view x[(Ns(td)+Nα(pd)+1):end]

    ds = @view f[1:Ns(td)]
    dα = @view f[(Ns(td)+1):(Ns(td)+Nα(pd))]
    dr = @view f[(Ns(td)+Nα(pd)+1):end]

    dynamics_function!(td, ds, s)
    input_function!(id, ds, r, u) # always call second, additive
    dynamics_function!(pd, dα, α, s)
    dynamics_function!(id, dr, s, r, u)
end

function combined_jump!(xₖ₊₁, xₖ, Δt, process::Process)
    @. xₖ₊₁[:] = xₖ[:]
end

function collocation_constraint!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, f!, Δt)
    Nx = length(xₖ)
    fₖ = zeros(eltype(r), Nx)
    fₖ₊₁ = zeros(eltype(r), Nx)
    fₘ = zeros(eltype(r), Nx)

    f!(fₖ, xₖ, uₖ)
    f!(fₖ₊₁, xₖ₊₁, uₖ₊₁)

    xₘ = 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    uₘ = 0.5 * (uₖ + uₖ₊₁)
    ẋₘ = (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    f!(fₘ, xₘ, uₘ)

    @. r[:] = fₘ - ẋₘ
end

function j_constraint!(r, xₖ, xₖ₊₁, fd!, Δt)
    Nx = length(xₖ)
    xp = zeros(eltype(xₖ), Nx)
    fd!(xp, xₖ, Δt[1])

    @. r[:] = xₖ₊₁ - xp
end

function constraints!(c, z, idx, colloc!, jump_constraint!, xᵢ; xf=nothing, Δt=nothing)
    # Collocation, Initial state, Final State
    Nc, Nkb = idx.Nc, idx.Nkb
    Nx, nu = idx.Nstates, idx.Nu
    free_time = isnothing(Δt)

    i = 0
    for cyc in 1:Nc
        if free_time
            Δt = @view z[idx.Δt[cyc]]
        end
        tc = @view z[idx.tc[cyc]]

        for k in 1:(Nkb-1)
            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            r = @view c[(i+1):(i+Nx)]
            i += Nx

            colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
        end

        if cyc < Nc
            xₖ = @view z[idx.x[cyc][Nk]]
            xₖ₊₁ = @view z[idx.x[cyc+1][1]]

            r = @view c[(i+1):(i+Nx)]
            i += Nx
            jump_constraint!(r, xₖ, xₖ₊₁, tc)
        end
    end

    @. c[(end-2Nx+1):(end-Nx)] = z[idx.x[1][1]] - xᵢ
    @. c[(end-Nx+1):end] = z[idx.x[Nc][Nkb]] - xf
    # c[end-Nx+1] = 0
end

function constraint_jacobian!(jac, z, idx, colloc!::Function, jump_constraint!::Function, sparsity_cache; Δt=nothing, prob=nothing)
    Nc, Nk = idx.Nc, idx.Nkb
    Nx = idx.Nstates
    nu = idx.Nu
    colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color, jc_∂xₖ_sparsity, jc_∂xₖ₊₁_sparsity, jc_∂Δt_sparsity, jc_∂xₖ_color, jc_∂xₖ₊₁_color, jc_∂Δt_color = sparsity_cache
    free_time = isnothing(Δt)

    colloc_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X), Δt, colorvec=colloc_∂Δt_color)
    colloc_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt), xₖ, colorvec=colloc_∂xₖ_color)
    colloc_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt), uₖ, colorvec=colloc_∂uₖ_color)
    colloc_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, X, uₖ₊₁, Δt), xₖ₊₁, colorvec=colloc_∂xₖ₊₁_color)
    colloc_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> colloc!(r, xₖ, uₖ, xₖ₊₁, X, Δt), uₖ₊₁, colorvec=colloc_∂uₖ₊₁_color)

    jc_∂xₖ(J, xₖ, xₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> jump_constraint!(r, X, xₖ₊₁, Δt), xₖ, colorvec=jc_∂xₖ_color)
    jc_∂xₖ₊₁(J, xₖ, xₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> jump_constraint!(r, xₖ, X, Δt), xₖ₊₁, colorvec=jc_∂xₖ₊₁_color)
    jc_∂Δt(J, xₖ, xₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> jump_constraint!(r, xₖ, xₖ₊₁, X), Δt, colorvec=jc_∂Δt_color)

    i = 1
    J∂xₖ = Float64.(sparse(colloc_∂xₖ_sparsity))
    J∂uₖ = Float64.(sparse(colloc_∂uₖ_sparsity))
    J∂xₖ₊₁ = Float64.(sparse(colloc_∂xₖ₊₁_sparsity))
    J∂uₖ₊₁ = Float64.(sparse(colloc_∂uₖ₊₁_sparsity))
    J∂Δt = Float64.(sparse(colloc_∂Δt_sparsity))

    Jc∂xₖ = Float64.(sparse(jc_∂xₖ_sparsity))
    Jc∂xₖ₊₁ = Float64.(sparse(jc_∂xₖ₊₁_sparsity))
    Jc∂Δt = Float64.(sparse(jc_∂Δt_sparsity))

    for cyc in 1:Nc
        if free_time
            Δt = @view z[idx.Δt[cyc]]
        end
        tc = @view z[idx.tc[cyc]]

        for k in 1:(Nk-1)
            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]

            if free_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
                _, _, vals = findnz(J∂Δt)
                jac[i:(i+length(vals)-1)] .= vals
                i += length(vals)
            end

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            # display(J∂xₖ)
            # res = zeros(Nx, Nx)
            # rp = zeros(Nx)
            # ForwardDiff.jacobian!(res, (r, X) -> colloc!(r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt), rp, xₖ)
            # display(res)
            r, c, vals = findnz(J∂xₖ)
            jac[i:(i+length(vals)-1)] .= vals
            i += length(vals)

            colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            # display(J∂uₖ)
            # res = zeros(Nx, nu)
            # rp = zeros(Nx)
            # ForwardDiff.jacobian!(res, (r, X) -> colloc!(r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt), rp, uₖ)
            # display(res)
            r, c, vals = findnz(J∂uₖ)
            jac[i:(i+length(vals)-1)] .= vals
            i += length(vals)

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            # display(J∂xₖ₊₁)
            # res = zeros(Nx, Nx)
            # rp = zeros(Nx)
            # ForwardDiff.jacobian!(res, (r, X) -> colloc!(r, xₖ, uₖ, X, uₖ₊₁, Δt), rp, xₖ₊₁)
            # display(res)
            _, _, vals = findnz(J∂xₖ₊₁)
            jac[i:(i+length(vals)-1)] .= vals
            i += length(vals)

            colloc_∂uₖ₊₁(J∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            # display(J∂uₖ₊₁)
            # res = zeros(Nx, nu)
            # rp = zeros(Nx)
            # ForwardDiff.jacobian!(res, (r, X) -> colloc!(r, xₖ, uₖ, xₖ₊₁, X, Δt), rp, uₖ₊₁)
            # display(res)
            _, _, vals = findnz(J∂uₖ₊₁)
            jac[i:(i+length(vals)-1)] .= vals
            i += length(vals)
        end

        if cyc < Nc
            xₖ = @view z[idx.x[cyc][Nk]]
            xₖ₊₁ = @view z[idx.x[cyc+1][1]]

            jc_∂xₖ(Jc∂xₖ, xₖ, xₖ₊₁, tc)
            _, _, vals = findnz(Jc∂xₖ)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            jc_∂xₖ₊₁(Jc∂xₖ₊₁, xₖ, xₖ₊₁, tc)
            _, _, vals = findnz(Jc∂xₖ₊₁)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)

            jc_∂Δt(Jc∂Δt, xₖ, xₖ₊₁, tc)
            _, _, vals = findnz(Jc∂Δt)
            jac[i:(i+length(vals))] .= vals
            i += length(vals)
        end
    end

    jac[(end-2Nx+1):(end-Nx)] .= 1
    jac[(end-Nx+1):(end)] .= 1
    # jac[end-Nx+1] = 0

    # res = zeros(idx.Nconstr, idx.Nz)
    # rp = zeros(idx.Nconstr)
    # ForwardDiff.jacobian!(res, prob.eval_constraint!, rp, z)
    # display(res)

    # rs = [r for (r,c) in prob.constraint_jacobian_sparsity]
    # cs = [c for (r,c) in prob.constraint_jacobian_sparsity]
    # display(Matrix(sparse(rs, cs, jac)))

    # @show jac
end

function constraint_jacobian_sparsity(idx, colloc!::Function, jump_constraint!::Function; Δt=nothing)
    Nx, Nu = idx.Nstates, idx.Nu
    Nk, Nc = idx.Nkb, idx.Nc
    Nconstr = idx.Nconstr
    free_time = isnothing(Δt)

    rd = ones(Symbolics.Num, Nx)
    xd1 = 2 * ones(Symbolics.Num, Nx)
    ud1 = 3 * ones(Symbolics.Num, Nu)
    xd2 = 4 * ones(Symbolics.Num, Nx)
    ud2 = 5 * ones(Symbolics.Num, Nu)
    Δtd = 6 * ones(Symbolics.Num, 1)

    colloc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, X, ud1, xd2, ud2, Δtd), rd, xd1)
    display(colloc_∂xₖ_sparsity)
    colloc_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd1, X, xd2, ud2, Δtd), rd, ud1)
    display(colloc_∂uₖ_sparsity)
    colloc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd1, ud1, X, ud2, Δtd), rd, xd2)
    colloc_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd1, ud1, xd2, X, Δtd), rd, ud2)
    colloc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> colloc!(r, xd1, ud1, xd2, ud2, X), rd, Δtd)
    if free_time
        display(colloc_∂Δt_sparsity)
    end

    jc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> jump_constraint!(r, X, xd2, Δtd), rd, xd1)
    jc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> jump_constraint!(r, xd1, X, Δtd), rd, xd2)
    jc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> jump_constraint!(r, xd1, xd2, X), rd, Δtd)

    colloc_∂uₖ_color = matrix_colors(Float64.(colloc_∂uₖ_sparsity))
    colloc_∂xₖ_color = matrix_colors(Float64.(colloc_∂xₖ_sparsity))
    colloc_∂xₖ₊₁_color = matrix_colors(Float64.(colloc_∂xₖ₊₁_sparsity))
    colloc_∂uₖ₊₁_color = matrix_colors(Float64.(colloc_∂uₖ₊₁_sparsity))
    colloc_∂Δt_color = matrix_colors(Float64.(colloc_∂Δt_sparsity))

    jc_∂xₖ_color = matrix_colors(Float64.(jc_∂xₖ_sparsity))
    jc_∂xₖ₊₁_color = matrix_colors(Float64.(jc_∂xₖ₊₁_sparsity))
    jc_∂Δt_color = matrix_colors(Float64.(jc_∂Δt_sparsity))

    NΔt = free_time ? 1 : 0
    row_offset = 0
    col_offset = 0
    rows = []
    cols = []
    for cyc in 1:Nc
        for k in 1:(Nk-1)
            if free_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(rows, r .+ row_offset)
                append!(cols, c)
            end
            col_offset = NΔt * cyc + (Nx + Nu) * (k - 1) + ((Nx + Nu) * (Nk - 1)) * (cyc - 1)

            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(colloc_∂uₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nu

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(colloc_∂uₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nu

            row_offset += Nx
        end

        if cyc < Nc
            col_offset = NΔt + (Nx + Nu) * (Nk - 1) + ((Nx + Nu) * (Nk - 1)) * (cyc - 1)
            r, c, _ = findnz(jc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(jc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(jc_∂Δt_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            row_offset += Nx
        end
    end

    append!(rows, collect((Nconstr-2Nx+1):(Nconstr-Nx)))
    append!(cols, collect(idx.x[1][1]))

    append!(rows, collect((Nconstr-Nx+1):(Nconstr)))
    append!(cols, collect(idx.x[Nc][Nk]))

    display(sparse(rows, cols, trues(length(cols))))
    total_structure = collect(zip(rows, cols))
    sparsity_cache = colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color, jc_∂xₖ_sparsity, jc_∂xₖ₊₁_sparsity, jc_∂Δt_sparsity, jc_∂xₖ_color, jc_∂xₖ₊₁_color, jc_∂Δt_color
    return total_structure, sparsity_cache
end


function MOI.eval_objective(prob::AdditiveProblem, z)
    return cost(prob.objective, z, prob.idx)
end

function MOI.eval_objective_gradient(prob::AdditiveProblem, grad_f, z)
    # display("Objective gradient")
    gradient(prob.objective, grad_f, z, prob.idx)
    # # if !isnothing(prob.gradient)
    # #     prob.objective_gradient!(grad_f, z)
    # # else

    # @time ForwardDiff.gradient(grad_f, (z) -> cost(prob.objective, z, prob.idx), z)
    # # end
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    # display("Constraint")
    prob.eval_constraint!(c, z)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    # display("Constraint jacobian")
    constraint_jacobian!(jac, z, prob.idx, prob.colloc!, prob.jump_constraint!, prob.sparsity_cache, Δt=prob.Δt, prob=prob)
end

MOI.features_available(prob::AdditiveProblem) = [:Grad, :Jac]
MOI.initialize(prob::AdditiveProblem, features) = nothing
MOI.jacobian_structure(prob::AdditiveProblem) = prob.constraint_jacobian_sparsity

function optimize_trajectory(problem::AdditiveProblem;
    X₀=nothing, U₀=nothing,
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    idx = problem.idx
    Nz, Nconstr = idx.Nz, idx.Nconstr
    process = problem.process
    Nkb, Nc = problem.Nkb, problem.Nc
    id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics

    # X₀ = range(problem.x₀, problem.x̄, length=Nkb)
    # @show size(X₀)
    if isnothing(z₀)
        z₀ = zeros(Nz)
        for c in 1:Nc
            if isnothing(problem.Δt)
                z₀[idx.Δt[c]] = 0.04
            end
            for k in 1:Nkb
                z₀[idx.x[c][k]] .= problem.x̄
                z₀[idx.u[c][k]] .= [300.0]
                z₀[idx.s[c][k][1]] = 600
            end
        end
    end



    c_l = zeros(Nconstr)
    c_u = zeros(Nconstr)
    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, Nz)

    # Set primal bounds and initial values
    for c in 1:Nc
        if isnothing(problem.Δt)
            Δtc = z[idx.Δt[c]]
            MOI.add_constraint(solver, Δtc, MOI.LessThan(0.1))
            MOI.add_constraint(solver, Δtc, MOI.GreaterThan(0.001))
        end

        for k in 1:Nkb

            uj = z[idx.u[c][k]]
            MOI.add_constraints(solver, uj, MOI.LessThan.(input_max(id)))
            MOI.add_constraints(solver, uj, MOI.GreaterThan.(input_min(id)))
            # for (j, idxs) in zip(1:Nu(id), idx.u[c][k])
            #     MOI.set(solver, MOI.VariablePrimalStart(), z[idxs], z₀[idx.u[c][k]][j])
            # end
            sj = z[idx.s[c][k]]
            MOI.add_constraints(solver, sj, MOI.LessThan.(state_max(td)))
            MOI.add_constraints(solver, sj, MOI.GreaterThan.(state_min(td)))
            # for (j, idxs) in zip(1:Ns(td), idx.s[c][k])

            #     MOI.set(solver, MOI.VariablePrimalStart(), z[idxs], z₀[idx.s[c][k]][j])
            # end
            αj = z[idx.α[c][k]]
            MOI.add_constraints(solver, αj, MOI.LessThan.(property_max(pd)))
            MOI.add_constraints(solver, αj, MOI.GreaterThan.(property_min(pd)))
            # for (j, idxs) in zip(1:Nα(pd), idx.α[c][k])
            #     MOI.set(solver, MOI.VariablePrimalStart(), z[idxs], problem.x₀[Ns(td)+j])
            # end
            rj = z[idx.r[c][k]]
            MOI.add_constraints(solver, rj, MOI.LessThan.(state_max(id)))
            MOI.add_constraints(solver, rj, MOI.GreaterThan.(state_min(id)))
            # for (j, idxs) in zip(1:Nr(id), idx.r[c][k])
            #     MOI.set(solver, MOI.VariablePrimalStart(), z[idxs], problem.x₀[Ns(td)+Nα(pd)+j])
            # end
        end

        for i in 1:length(z)
            MOI.set(solver, MOI.VariablePrimalStart(), z[i], z₀[i])
        end
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    X = [[result[idx.x[c][k]] for k in 1:Nkb] for c in 1:Nc]
    U = [[result[idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]
    Δt = nothing
    if isnothing(problem.Δt)
        Δt = [result[idx.Δt[c]] for c in 1:Nc]
    end
    tc = [result[idx.tc[c]] for c in 1:Nc]

    return result, X, U, Δt, tc
end

end