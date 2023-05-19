module ADDOPT

using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
using Symbolics, SparseArrays, SparseDiffTools
using SparseArrays: findnz
const MOI = MathOptInterface

using InteractiveUtils
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
include("rollout.jl")

fₖ_cache = Dict{DataType,Any}()
fₖ₊₁_cache = Dict{DataType,Any}()
fₘ_cache = Dict{DataType,Any}()
xₘ_cache = Dict{DataType,Any}()
uₘ_cache = Dict{DataType,Any}()
ẋₘ_cache = Dict{DataType,Any}()

struct AdditiveProblem{OB<:Objective, ID<:InputDynamics, TD<:TransferDynamics, PD<:PropertyDynamics} <: MOI.AbstractNLPEvaluator
    process::Process{ID, TD, PD}

    objective::OB
    # objective_gradient!::Union{Function,Nothing}
    # hessian::Union{Function,Matrix,Nothing}

    # Nx::Int
    # Ny::Int
    # l::Real

    Δt::Union{Real,Nothing}
    Nkb::Int # Number of knots per build cycle
    # Nkc::Int # Number of knots per cooling cycle
    Nc::Int  # Number of cycles

    #eval_constraint!::Function
    constraint_jacobian_sparsity

    idx
    sparsity_cache
    #colloc!
    jump_constraint!

    x₀::Vector{Float64}
    x̄::Vector{Float64}

    function AdditiveProblem(process::Process{ID, TD, PD}, objective::OB, #Nx, Ny, l, 
        Nkb, Nc, x₀; x̄=nothing, Δt=nothing) where {OB<:Objective, ID<:InputDynamics, TD<:TransferDynamics, PD<:PropertyDynamics}
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        free_time = isnothing(Δt)
        idx = generate_z_indices(Nkb, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), free_time=free_time)

        f!(ẋ, x, u) = combined_dynamics!(ẋ, x, u, process)
        fd!(xₖ₊₁, xₖ, Δt) = combined_jump!(xₖ₊₁, xₖ, Δt, process)

        #colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = collocation_constraint!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, f!, Δt)#; fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ), fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ
        jump_constraint!(r, xₖ, xₖ₊₁, Δt) = j_constraint!(r, xₖ, xₖ₊₁, fd!, Δt)
        # eval_constraint!(c, z) = constraints!(c, z, idx, colloc!, jump_constraint!, x₀; xf=x̄, Δt=Δt)

        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, process, jump_constraint!, Δt=Δt)

        new{OB, ID, TD, PD}(process, objective, Δt, Nkb, Nc,
            #eval_constraint!, 
            con_jacobian_sparsity,
            idx, sparsity_cache, #colloc!, 
            jump_constraint!, 
            x₀, x̄)#Nx, Ny, l,
    end
end

function generate_z_indices(Nkb, Nc, Nu, Nr, Ns, Nα; free_time=false)
    NΔt = free_time ? 1 : 0
    Nstates = (Ns + Nα)#Nx * Ny * 
    Nknotvals = Nstates + Nr + Nu + NΔt
    Npercycle = (Nknotvals * Nkb + 1)
    Nz = Npercycle * Nc #+ Nstates # Last term is for state after final cooling

    offset(k, c) = Nknotvals * (k - 1) + Npercycle * (c - 1)

    s = [[(1:Ns) .+ offset(k, c) for k in 1:Nkb] for c in 1:Nc]                             # z[s[c][k]] gives s_(c,k), vector
    α = [[((Ns+1):(Nstates)) .+ offset(k, c) for k in 1:Nkb] for c in 1:Nc]                 # z[α[c][k]] gives α_(c,k), vector
    r = [[((Nstates+1):(Nstates+Nr)) .+ offset(k, c) for k in 1:Nkb] for c in 1:Nc]         # z[r[c][k]] gives r_(c,k), vector
    u = [[((Nstates+Nr+1):(Nstates+Nr+Nu)) .+ offset(k, c) for k in 1:Nkb] for c in 1:Nc]   # z[u[c][k]] gives u_(c,k), vector
    Δt = free_time ? [[(Nstates + Nr + Nu + 1) + offset(k, c) for k in 1:Nkb] for c in 1:Nc] : nothing # z[Δt[c][k]] gives Δt_(c,k), scalar ##vector singleton
    x = [[(1:(Nstates+Nr)) .+ offset(k, c) for k in 1:Nkb] for c in 1:Nc]                   # z[x[c][k]] gives x_(c,k), vector
    tc = [Npercycle * c for c in 1:Nc]                                                      # z[tc[c]] gives tc_c, scalar

    Nconstr = (Nstates + Nr) * (Nkb - 1) * Nc + (Nstates + Nr) * 2

    return (Nz=Nz, Nstates=(Nstates + Nr), s=s, α=α, r=r, u=u, x=x, Δt=Δt, tc=tc, Nconstr=Nconstr, Nkb=Nkb, Nc=Nc, Nu=Nu)
end

function combined_dynamics!(f, x, u, process::Process{ID, TD, PD}) where {ID, TD, PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics
    # Ns, Nα(pd) = Ns(td)(td), Nα(pd)(pd)
    # Nr, Nu = Nr(id), Nu(id)

    s = view(x, 1:Ns(td))
    α = view(x, (Ns(td)+1):(Ns(td)+Nα(pd)))
    r = view(x, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))
 
    ds = view(f, 1:Ns(td))
    dα = view(f, (Ns(td)+1):(Ns(td)+Nα(pd)))
    dr = view(f, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))
 
    dynamics_function!(td, ds, s)
    input_function!(id, ds, r, u) # always call second, additive
    dynamics_function!(pd, td, dα, α, s)
    dynamics_function!(id, dr, s, r, u)
end

function combined_jump!(xₖ₊₁, xₖ, Δt, process::Process)
    @. xₖ₊₁[:] = xₖ[:]
end

function collocation_constraint!(process::Process{ID, TD, PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt; fₖ_cache::Dict{DataType,Any}=fₖ_cache, fₖ₊₁_cache::Dict{DataType,Any}=fₖ₊₁_cache, fₘ_cache::Dict{DataType,Any}=fₘ_cache, xₘ_cache::Dict{DataType,Any}=xₘ_cache, uₘ_cache::Dict{DataType,Any}=uₘ_cache, ẋₘ_cache::Dict{DataType,Any}=ẋₘ_cache) where {T, ID, TD, PD}
    Nx = length(xₖ)
    nu = length(uₖ)

    # get!(cache, T) do
    #     AbstractVector{T}(N)
    # end::AbstractVector{T}

    fₖ = get!(fₖ_cache, T) do
        zeros(T, Nx)
    end::Vector{T}
    fₖ₊₁ = get!(fₖ₊₁_cache, T) do
        zeros(T, Nx)
    end::Vector{T}
    fₘ = get!(fₘ_cache, T) do
        zeros(T, Nx)
    end::Vector{T}
    xₘ = get!(xₘ_cache, T) do
        zeros(T, Nx)
    end::Vector{T}
    uₘ = get!(uₘ_cache, T) do
        zeros(T, nu)
    end::Vector{T}
    ẋₘ = get!(ẋₘ_cache, T) do
        zeros(T, Nx)
    end::Vector{T}


    # if isnothing(fₖ)
    # fₖ = zeros(eltype(r), Nx)
    # fₖ₊₁ = zeros(eltype(r), Nx)
    # fₘ = zeros(eltype(r), Nx)
    # xₘ = zeros(eltype(r), Nx)
    # uₘ = zeros(eltype(r), nu)
    # ẋₘ = zeros(eltype(r), Nx)
    # end

    combined_dynamics!(fₖ, xₖ, uₖ, process)
    combined_dynamics!(fₖ₊₁, xₖ₊₁, uₖ₊₁, process)

    xₘ .= @. 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    uₘ .= @. 0.5 * (uₖ + uₖ₊₁)
    ẋₘ .= @. (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    combined_dynamics!(fₘ, xₘ, uₘ, process)

    r[:] .= fₘ .- ẋₘ
end

function j_constraint!(r, xₖ, xₖ₊₁, fd!, Δt)
    Nx = length(xₖ)
    xp = zeros(eltype(xₖ), Nx)
    fd!(xp, xₖ, Δt[1])

    @. r[:] = xₖ₊₁ - xp
end

function constraints!(process::Process, c, z, idx, jump_constraint!, xᵢ; xf=nothing, Δt=nothing)
    # Collocation, Initial state, Final State
    Nc, Nkb = idx.Nc, idx.Nkb
    Nx, nu = idx.Nstates, idx.Nu
    free_time = isnothing(Δt)
    # fₖ = zeros(eltype(c), Nx)
    # fₖ₊₁ = zeros(eltype(c), Nx)
    # fₘ = zeros(eltype(c), Nx)
    # xₘ = zeros(eltype(c), Nx)
    # uₘ = zeros(eltype(c), nu)
    # ẋₘ = zeros(eltype(c), Nx)

    i = 0
    for cyc in 1:Nc
        tc = @view z[idx.tc[cyc]]

        for k in 1:(Nkb-1)
            if free_time
                Δt = @view z[idx.Δt[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            r = @view c[(i+1):(i+Nx)]
            i += Nx

            collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            # colloc!(r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)#, fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ)
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

function constraint_jacobian!(process::Process, jac, z, idx, jump_constraint!::Function, sparsity_cache; Δt=nothing, prob=nothing)
    Nc, Nk = idx.Nc, idx.Nkb
    Nx = idx.Nstates
    nu = idx.Nu
    colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color, jc_∂xₖ_sparsity, jc_∂xₖ₊₁_sparsity, jc_∂Δt_sparsity, jc_∂xₖ_color, jc_∂xₖ₊₁_color, jc_∂Δt_color = sparsity_cache
    free_time = isnothing(Δt)

    # fₖ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, Nx)
    # fₖ₊₁ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, Nx)
    # fₘ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, Nx)
    # xₘ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, Nx)
    # uₘ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, nu)
    # ẋₘ = zeros(ForwardDiff.Dual{T,V,N} where {T,V,N}, Nx)

    # cfg = ForwardDiff.JacobianConfig(f!, y, x, ForwardDiff.Chunk{3}())
    colloc_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X), Δt, colorvec=colloc_∂Δt_color)
    colloc_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt), xₖ, colorvec=colloc_∂xₖ_color)
    colloc_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt), uₖ, colorvec=colloc_∂uₖ_color)
    colloc_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, X, uₖ₊₁, Δt), xₖ₊₁, colorvec=colloc_∂xₖ₊₁_color)
    colloc_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, X, Δt), uₖ₊₁, colorvec=colloc_∂uₖ₊₁_color)
    #     , fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ
    # , fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ
    # , fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ
    # , fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ
    # , fₖ, fₖ₊₁, fₘ, xₘ, uₘ, ẋₘ

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
        tc = @view z[idx.tc[cyc]]

        for k in 1:(Nk-1)
            if free_time
                Δt = @view z[idx.Δt[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]

            # @show eltype(xₖ  )
            # @show eltype(uₖ  )
            # @show eltype(xₖ₊₁)
            # @show eltype(uₖ₊₁)


            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂uₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ₊₁(J∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt)
            _, _, vals = findnz(J∂uₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
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

function constraint_jacobian_sparsity(idx, process::Process, jump_constraint!::Function; Δt=nothing)
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

    # fₖ = zeros(Symbolics.Num, Nx)
    # fₖ₊₁ = zeros(Symbolics.Num, Nx)
    # fₘ = zeros(Symbolics.Num, Nx)
    # xₘ = zeros(Symbolics.Num, Nx)
    # uₘ = zeros(Symbolics.Num, Nu)
    # ẋₘ = zeros(Symbolics.Num, Nx)

    colloc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, X, ud1, xd2, ud2, Δtd), rd, xd1)
    display(colloc_∂xₖ_sparsity)
    colloc_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, X, xd2, ud2, Δtd), rd, ud1)
    display(colloc_∂uₖ_sparsity)
    colloc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, X, ud2, Δtd), rd, xd2)
    colloc_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, xd2, X, Δtd), rd, ud2)
    colloc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, xd2, ud2, X), rd, Δtd)
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
            col_offset = (Nx + Nu + NΔt) * (k - 1) + ((Nx + Nu + NΔt) * (Nk - 1)) * (cyc - 1)

            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nx

            r, c, _ = findnz(colloc_∂uₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ col_offset)
            col_offset += Nu

            if free_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(rows, r .+ row_offset)
                append!(cols, c .+ col_offset)
                col_offset += NΔt
            end

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
    gradient(prob.objective, grad_f, z, prob.idx)
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    constraints!(prob.process, c, z, prob.idx, prob.jump_constraint!, prob.x₀, xf=prob.x̄, Δt=prob.Δt)
    # prob.eval_constraint!(c, z)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.jump_constraint!, prob.sparsity_cache, Δt=prob.Δt, prob=prob)
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

            for k in 1:Nkb
                if isnothing(problem.Δt)
                    z₀[idx.Δt[c][k]] = 0.04
                end
                z₀[idx.x[c][k]] .= problem.x̄
                z₀[idx.u[c][k]] .= [300.0]
                z₀[idx.s[c][k][1]] = 600
            end
        end
    end

    ct = zeros(Nconstr)
    gt = zeros(Nz)
    jt = zeros(length(problem.constraint_jacobian_sparsity))
    display("Checking objective function...")
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    display("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    display("Checking constraint function...")
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    display("Checking constraint jacobian...")
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)


    c_l = zeros(Nconstr)
    c_u = zeros(Nconstr)
    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, Nz)

    # Set primal bounds and initial values
    for c in 1:Nc
        for k in 1:Nkb
            if isnothing(problem.Δt)
                Δtc = z[idx.Δt[c][k]]
                MOI.add_constraint(solver, Δtc, MOI.LessThan(0.1))
                MOI.add_constraint(solver, Δtc, MOI.GreaterThan(0.001))
            end

            uj = z[idx.u[c][k]]
            MOI.add_constraints(solver, uj, MOI.LessThan.(input_max(id)))
            MOI.add_constraints(solver, uj, MOI.GreaterThan.(input_min(id)))

            sj = z[idx.s[c][k]]
            MOI.add_constraints(solver, sj, MOI.LessThan.(state_max(td)))
            MOI.add_constraints(solver, sj, MOI.GreaterThan.(state_min(td)))

            αj = z[idx.α[c][k]]
            MOI.add_constraints(solver, αj, MOI.LessThan.(property_max(pd)))
            MOI.add_constraints(solver, αj, MOI.GreaterThan.(property_min(pd)))

            rj = z[idx.r[c][k]]
            MOI.add_constraints(solver, rj, MOI.LessThan.(state_max(id)))
            MOI.add_constraints(solver, rj, MOI.GreaterThan.(state_min(id)))
        end

        for i in 1:lastindex(z)
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
        Δt = [[result[idx.Δt[c][k]] for k in 1:Nkb] for c in 1:Nc]
    end
    tc = [result[idx.tc[c]] for c in 1:Nc]

    return result, X, U, Δt, tc
end

end