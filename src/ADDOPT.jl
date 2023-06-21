module ADDOPT

using PrettyTables
@time using MathOptInterface, Ipopt
@time using LinearAlgebra, ForwardDiff
@time using Symbolics, SparseArrays, SparseDiffTools
@time using SparseArrays: findnz
const MOI = MathOptInterface

abstract type Dynamics end

abstract type InputDynamics <: Dynamics end
abstract type TransferDynamics <: Dynamics end
abstract type PropertyDynamics <: Dynamics end

Nu(id::InputDynamics) = 0
Nr(id::InputDynamics) = 0
Nc_ineq(id::InputDynamics) = 0
Nc_eq(id::InputDynamics) = 0
Nα(pd::PropertyDynamics) = 0
Ns(td::TransferDynamics) = 0

function equality_constraint!(id::InputDynamics, c, r, u, t)
end

function inequality_constraint!(id::InputDynamics, c, r, u, t)
end

ineq_min(id::InputDynamics) = []
ineq_max(id::InputDynamics) = []

include("processes.jl")
include("objectives.jl")
include("rollout.jl")
include("initial_guess.jl")
include("visualization.jl")

fₖ_cache = Dict{DataType,Any}()
fₖ₊₁_cache = Dict{DataType,Any}()
fₘ_cache = Dict{DataType,Any}()
xₘ_cache = Dict{DataType,Any}()
ẋₘ_cache = Dict{DataType,Any}()

struct AdditiveProblem{OB<:Objective,ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics} <: MOI.AbstractNLPEvaluator
    process::Process{ID,TD,PD}

    objective::OB

    Δtb::Union{Real,Nothing}
    Δtc::Union{Real,Nothing}
    Nkb::Int # Number of knots per build cycle
    Nkc::Int # Number of knots per cooling cycle
    Nc::Int  # Number of cycles

    constraint_jacobian_sparsity

    idx
    sparsity_cache

    x₀::Vector{Float64}
    x̄::Vector{Float64}

    final_constraint::Bool

    function AdditiveProblem(process::Process{ID,TD,PD}, objective::OB,
        Nkb, Nkc, Nc, x₀; x̄=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false) where {OB<:Objective,ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics}
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        idx = generate_z_indices(Nkb, Nkc, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), Nc_eq(id), Nc_ineq(id), Δtb, Δtc, final_constraint)

        println("Preparing sparsity")
        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, process; Δtb=Δtb, Δtc=Δtc, final_constraint=final_constraint)
        println("Done with sparsity")

        new{OB,ID,TD,PD}(process, objective, Δtb, Δtc, Nkb, Nkc, Nc,
            con_jacobian_sparsity,
            idx, sparsity_cache,
            x₀, x̄,
            final_constraint)
    end
end

function generate_z_indices(Nkb, Nkc, Nc, Nu, Nr, Ns, Nα, Neq, Nineq, Δtb, Δtc, final_constraint)
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    Nstates = Ns + Nα + Nr
    StatesPerCycle = Nstates * (Nkb + Nkc)
    Nstates_total = StatesPerCycle * Nc

    InputsPerCycle = Nu * Nkb
    Nu_total = InputsPerCycle * Nc

    NΔtb = free_build_time ? 1 : 0
    ΔtbPerCycle = NΔtb * Nkb
    NΔtb_total = ΔtbPerCycle * Nc

    NΔtc = free_cool_time ? 1 : 0
    ΔtcPerCycle = NΔtc * Nkc
    NΔtc_total = ΔtcPerCycle * Nc

    Nz = Nstates_total + Nu_total + NΔtb_total + NΔtc_total

    x = [[(1:Nstates) .+ (Nstates * (k - 1) + StatesPerCycle * (c - 1)) for k in 1:(Nkb+Nkc)] for c in 1:Nc]
    u = [[(1:Nu) .+ (Nu * (k - 1) + InputsPerCycle * (c - 1) + Nstates_total) for k in 1:Nkb] for c in 1:Nc]
    Δtb = free_build_time ? [[k + (Nstates_total + Nu_total + ΔtbPerCycle * (c - 1)) for k in 1:Nkb] for c in 1:Nc] : nothing # z[Δtb[c][k]] gives Δtb_(c,k), scalar
    Δtc = free_cool_time ? [vcat([0 for k in 1:Nkb], [k + (Nstates_total + Nu_total + NΔtb_total + ΔtcPerCycle * (c - 1)) for k in 1:Nkc]) for c in 1:Nc] : nothing

    Nconstr = (Nstates + Neq + Nineq) * ((Nkb + Nkc) * Nc - 1) + (final_constraint ? 2Nstates : Nstates)

    return (Nz=Nz, Nstates=Nstates, u=u, x=x, Δtb=Δtb, Δtc=Δtc, Nconstr=Nconstr, Nkb=Nkb, Nkc=Nkc, Nc=Nc, Nu=Nu, Neq=Neq, Nineq=Nineq)
end

function combined_dynamics!(f, x, u, process::Process{ID,TD,PD}, t) where {ID,TD,PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics

    s = view(x, 1:Ns(td))
    α = view(x, (Ns(td)+1):(Ns(td)+Nα(pd)))
    r = view(x, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    ds = view(f, 1:Ns(td))
    dα = view(f, (Ns(td)+1):(Ns(td)+Nα(pd)))
    dr = view(f, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    dynamics_function!(td, ds, s, t)
    input_function!(id, ds, r, u, t) # always call second, additive
    dynamics_function!(pd, td, dα, α, s, t)
    dynamics_function!(id, dr, s, r, u, t)
end

function collocation_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, Δt, t, Neq, Nineq; fₖ_cache::Dict{DataType,Any}=fₖ_cache, fₖ₊₁_cache::Dict{DataType,Any}=fₖ₊₁_cache, fₘ_cache::Dict{DataType,Any}=fₘ_cache, xₘ_cache::Dict{DataType,Any}=xₘ_cache, ẋₘ_cache::Dict{DataType,Any}=ẋₘ_cache) where {T,ID,TD,PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics
    Nx = length(xₖ)
    nu = length(uₖ)

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
    ẋₘ = get!(ẋₘ_cache, T) do
        zeros(T, Nx)
    end::Vector{T}

    combined_dynamics!(fₖ, xₖ, uₖ, process, t)
    combined_dynamics!(fₖ₊₁, xₖ₊₁, uₖ, process, t + Δt[1])

    xₘ .= @. 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    ẋₘ .= @. (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    combined_dynamics!(fₘ, xₘ, uₖ, process, t + Δt[1] / 2.0)


    r[1:Nx] .= fₘ .- ẋₘ

    ri = view(xₖ, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))
    equality_constraint!(id, view(r, (Nx+1):(Nx+Neq)), ri, uₖ, t)
    inequality_constraint!(id, view(r, (Nx+Neq+1):(Nx+Neq+Nineq)), ri, uₖ, t)
end

function constraints!(process::Process, c, z, idx, xᵢ; xf=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false)
    # Collocation, Initial state, Final State
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Nx, nu = idx.Nstates, idx.Nu
    Neq, Nineq = idx.Neq, idx.Nineq
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)
    Ncon = Nx + Neq + Nineq

    i = 0
    t = 0.0
    for cyc in 1:Nc
        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end

            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]
            r = @view c[(i+1):(i+Ncon)]

            collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, Δtc, t, Neq, Nineq)
            i += Ncon

            t += Δtc
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            r = @view c[(i+1):(i+Ncon)]
            i += Ncon

            collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, Δtb, t, Neq, Nineq)
            t += Δtb
        end

        k = Nkb
        if free_build_time
            Δtb = @view z[idx.Δtb[cyc][k]]
        end

        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)
        r = @view c[(i+1):(i+Ncon)]
        i += Ncon

        collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, Δtb, t, Neq, Nineq)
        t += Δtb

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            r = @view c[(i+1):(i+Ncon)]
            i += Ncon

            collocation_constraint!(process, r, xₖ, uᵢ, xₖ₊₁, Δtc, t, Neq, Nineq)
            t += Δtc
        end

    end

    if final_constraint
        @. c[(end-2Nx+1):(end-Nx)] = z[idx.x[Nc][Nkb+Nkc]] - xf
    end
    @. c[(end-Nx+1):end] = z[idx.x[1][1]] - xᵢ
end

function constraint_jacobian!(process::Process, jac, z, idx, sparsity_cache; Δtb=nothing, Δtc=nothing, prob=nothing, final_constraint=false)
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Neq, Nineq = idx.Neq, idx.Nineq
    Nx = idx.Nstates
    nu = idx.Nu
    Ncon = Nx + Neq + Nineq
    colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂Δt_color = sparsity_cache
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    colloc_∂Δt(J, xₖ, uₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, X, t, Neq, Nineq), Δt, colorvec=colloc_∂Δt_color)
    colloc_∂xₖ(J, xₖ, uₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, X, uₖ, xₖ₊₁, Δt, t, Neq, Nineq), xₖ, colorvec=colloc_∂xₖ_color)
    colloc_∂uₖ(J, xₖ, uₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, X, xₖ₊₁, Δt, t, Neq, Nineq), uₖ, colorvec=colloc_∂uₖ_color)
    colloc_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, X, Δt, t, Neq, Nineq), xₖ₊₁, colorvec=colloc_∂xₖ₊₁_color)

    i = 1
    J∂xₖ = Float64.(sparse(colloc_∂xₖ_sparsity))
    J∂uₖ = Float64.(sparse(colloc_∂uₖ_sparsity))
    J∂xₖ₊₁ = Float64.(sparse(colloc_∂xₖ₊₁_sparsity))
    J∂Δt = Float64.(sparse(colloc_∂Δt_sparsity))
    t = 0.0

    for cyc in 1:Nc

        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end
            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, Δtc, t)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
            t += Δtc
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, Δtb, t)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, Δtb, t)
            _, _, vals = findnz(J∂uₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_build_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, Δtb, t)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, Δtb, t)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
            t += Δtb
        end

        k = Nkb
        if free_build_time
            Δtb = @view z[idx.Δtb[cyc][k]]
        end
        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)

        colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, Δtb, t)
        _, _, vals = findnz(J∂xₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, Δtb, t)
        _, _, vals = findnz(J∂uₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        if free_build_time
            colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, Δtb, t)
            _, _, vals = findnz(J∂Δt)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end

        colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, Δtb, t)
        _, _, vals = findnz(J∂xₖ₊₁)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)
        t += Δtb

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uᵢ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                colloc_∂Δt(J∂Δt, xₖ, uᵢ, xₖ₊₁, Δtc, t)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uᵢ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
            t += Δtc
        end
    end

    if final_constraint
        jac[(end-2Nx+1):(end-Nx)] .= 1
    end
    jac[(end-Nx+1):(end)] .= 1

    # res = zeros(idx.Nconstr, idx.Nz)
    # rp = zeros(idx.Nconstr)
    # ForwardDiff.jacobian!(res, (r,z) -> constraints!(process, r, z, idx, prob.x₀; xf=prob.x̄, Δtb=Δtb, Δtc=Δtc), rp, z)
    # println("reference")#tf = borderless,
    # pretty_table(res,  noheader = true, crop = :none, formatters = ft_printf("%3.1e"))
    # # show(stdout, "text/plain", res)
    # # display(sparse(res))

    # rs = [r for (r,c) in prob.constraint_jacobian_sparsity]
    # cs = [c for (r,c) in prob.constraint_jacobian_sparsity]
    # println("actual")
    # pretty_table(sparse(rs, cs, jac), noheader = true, crop = :none, formatters = ft_printf("%3.1e"))
    # # show(stdout, "text/plain", Matrix(sparse(rs, cs, jac)))
    # # display(norm(sparse(rs, cs, jac)-sparse(res)))
end

function constraint_jacobian_sparsity(idx, process::Process; Δtb=nothing, Δtc=nothing, final_constraint=false)
    Nx, Nu = idx.Nstates, idx.Nu
    Nkb, Nkc, Nc = idx.Nkb, idx.Nkc, idx.Nc
    Nconstr = idx.Nconstr
    Neq, Nineq = idx.Neq, idx.Nineq
    Ncon = Nx + Neq + Nineq
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    rd = ones(Symbolics.Num, Ncon)
    xd1 = 0.002 * ones(Symbolics.Num, Nx)
    ud1 = 0.003 * ones(Symbolics.Num, Nu)
    xd2 = 0.004 * ones(Symbolics.Num, Nx)
    Δtd = 0.006 * ones(Symbolics.Num, 1)
    t = 0.0

    println("Computing sparsity xₖ")
    colloc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, X, ud1, xd2, Δtd, t, Neq, Nineq), rd, xd1)
    display(colloc_∂xₖ_sparsity)
    println("Computing sparsity uₖ")
    colloc_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, X, xd2, Δtd, t, Neq, Nineq), rd, ud1)
    display(colloc_∂uₖ_sparsity)
    colloc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, X, Δtd, t, Neq, Nineq), rd, xd2)
    colloc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, xd2, X, t, Neq, Nineq), rd, Δtd)
    if free_build_time || free_cool_time
        display(colloc_∂Δt_sparsity)
    end

    colloc_∂uₖ_color = matrix_colors(Float64.(colloc_∂uₖ_sparsity))
    colloc_∂xₖ_color = matrix_colors(Float64.(colloc_∂xₖ_sparsity))
    colloc_∂xₖ₊₁_color = matrix_colors(Float64.(colloc_∂xₖ₊₁_sparsity))
    colloc_∂Δt_color = matrix_colors(Float64.(colloc_∂Δt_sparsity))

    NΔtb = free_build_time ? 1 : 0
    NΔtc = free_cool_time ? 1 : 0
    row_offset = 0
    # rows = []
    # cols = []
    total_structure = Vector{Tuple{Int,Int}}()
    println("Entering loop")
    for cyc in 1:Nc
        if cyc > 1
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc-1][Nkb+Nkc][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc-1][Nkb+Nkc][1] .- 1)))
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][1][1] .- 1)))

            row_offset += Ncon
        end

        for k in 1:(Nkb-1)
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

            r, c, _ = findnz(colloc_∂uₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.u[cyc][k][1] .- 1)))

            if free_build_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc][k][1] .- 1)))
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

            row_offset += Ncon
        end

        k = Nkb
        r, c, _ = findnz(colloc_∂xₖ_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

        r, c, _ = findnz(colloc_∂uₖ_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.u[cyc][k][1] .- 1)))

        if free_build_time
            r, c, _ = findnz(colloc_∂Δt_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc][k][1] .- 1)))
        end

        r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

        row_offset += Ncon

        for k in (Nkb+1):(Nkb+Nkc-1)
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtc[cyc][k][1] .- 1)))
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

            row_offset += Ncon
        end

    end
    println("Finished loop")

    if final_constraint
        append!(total_structure, collect(zip(collect((Nconstr-2Nx+1):(Nconstr-Nx)), collect(idx.x[Nc][Nkb+Nkc]))))
    end

    append!(total_structure, collect(zip(collect((Nconstr-Nx+1):(Nconstr)), collect(idx.x[1][1]))))

    # display(sparse(rows, cols, trues(length(cols))))
    # total_structure = collect(zip(rows, cols))
    sparsity_cache = colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂Δt_color
    return total_structure, sparsity_cache
end


function MOI.eval_objective(prob::AdditiveProblem, z)
    return cost(prob.objective, z, prob.idx)
end

function MOI.eval_objective_gradient(prob::AdditiveProblem, grad_f, z)
    gradient(prob.objective, grad_f, z, prob.idx)
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    constraints!(prob.process, c, z, prob.idx, prob.x₀, xf=prob.x̄, Δtb=prob.Δtb, Δtc=prob.Δtc, final_constraint=prob.final_constraint)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    # @show [[z[prob.idx.u[c][k]] for k in 1:prob.idx.Nkb] for c in 1:prob.idx.Nc] 
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.sparsity_cache, Δtb=prob.Δtb, Δtc=prob.Δtc, prob=prob, final_constraint=prob.final_constraint)
end

function MOI.hessian_lagrangian_structure(prob::AdditiveProblem)
    return hessian_structure(prob.objective, prob.idx)
end

function MOI.eval_hessian_lagrangian(prob::AdditiveProblem, H, z, σ, μ)
    hessian_values(prob.objective, prob.idx, H)
    H .*= σ
end

MOI.features_available(prob::AdditiveProblem) = [:Grad, :Jac, :Hess]
MOI.initialize(prob::AdditiveProblem, features) = nothing
MOI.jacobian_structure(prob::AdditiveProblem) = prob.constraint_jacobian_sparsity

function optimize_trajectory(problem::AdditiveProblem;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing, λ₀=nothing, ug=nothing, xg=nothing)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    # solver.options["warm_start_init_point"] = "yes"
    # solver.options["warm_start_bound_push"] = 1e-9
    # solver.options["warm_start_bound_frac"] = 1e-9
    # solver.options["warm_start_slack_bound_frac"] = 1e-9
    # solver.options["warm_start_slack_bound_push"] = 1e-9
    # solver.options["warm_start_mult_bound_push"] = 1e-9


    idx = problem.idx
    Nz, Nconstr = idx.Nz, idx.Nconstr
    Nx, Neq, Nineq = idx.Nstates, idx.Neq, idx.Nineq
    process = problem.process
    Nkb, Nkc, Nc = problem.Nkb, idx.Nkc, problem.Nc
    id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics

    if isnothing(z₀)
        z₀ = zeros(Nz)
        for c in 1:Nc

            for k in 1:Nkb
                if isnothing(problem.Δtb)
                    z₀[idx.Δtb[c][k]] = 0.04
                end
                z₀[idx.x[c][k]] .= isnothing(xg) ? problem.x̄ : xg #.+ randn(Nx)
                z₀[idx.u[c][k]] .= isnothing(ug) ? input_min(id) .+ randn(Nu(id)) : ug#(input_min(id) .+ input_max(id)) ./ 2
            end

            for k in (Nkb+1):(Nkb+Nkc)
                if isnothing(problem.Δtc)
                    z₀[idx.Δtc[c][k]] = 0.04
                end
                z₀[idx.x[c][k]] .= problem.x̄
            end
        end
    end
    @show norm(z₀)
    @show isnothing(problem.Δtb)
    @show isnothing(problem.Δtc)

    ct = zeros(Nconstr)
    gt = zeros(Nz)
    jt = zeros(length(problem.constraint_jacobian_sparsity))
    println("Checking objective function...")
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @show MOI.eval_objective(problem, z₀)
    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    println("Checking constraint function...")
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @show maximum(abs.(ct))
    println("Checking constraint jacobian...")
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)

    c_l = vcat(repeat([zeros(Nx + Neq); ineq_min(id)], ((Nkb + Nkc) * Nc - 1)), zeros(problem.final_constraint ? 2Nx : Nx))
    c_u = vcat(repeat([zeros(Nx + Neq); ineq_max(id)], ((Nkb + Nkc) * Nc - 1)), zeros(problem.final_constraint ? 2Nx : Nx))

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, Nz)
    x_min = [state_min(td)
        property_min(pd)
        state_min(id)]
    x_max = [state_max(td)
        property_max(pd)
        state_max(id)]

    # Set primal bounds and initial values
    for c in 1:Nc
        for k in 1:Nkb
            if isnothing(problem.Δtb)
                Δtb = z[idx.Δtb[c][k]]
                MOI.add_constraint(solver, Δtb, MOI.LessThan(0.1))
                MOI.add_constraint(solver, Δtb, MOI.GreaterThan(0.001))
            end

            uj = z[idx.u[c][k]]
            MOI.add_constraints(solver, uj, MOI.LessThan.(input_max(id)))
            MOI.add_constraints(solver, uj, MOI.GreaterThan.(input_min(id)))

            xj = z[idx.x[c][k]]
            MOI.add_constraints(solver, xj, MOI.LessThan.(x_max))
            MOI.add_constraints(solver, xj, MOI.GreaterThan.(x_min))
        end

        for k in (Nkb+1):(Nkb+Nkc)
            if isnothing(problem.Δtc)
                Δtc = z[idx.Δtc[c][k]]
                MOI.add_constraint(solver, Δtc, MOI.LessThan(0.1))
                MOI.add_constraint(solver, Δtc, MOI.GreaterThan(0.001))
            end

            xj = z[idx.x[c][k]]
            MOI.add_constraints(solver, xj, MOI.LessThan.(x_max))
            MOI.add_constraints(solver, xj, MOI.GreaterThan.(x_min))
        end

    end

    for i in 1:lastindex(z₀)
        MOI.set(solver, MOI.VariablePrimalStart(), z[i], z₀[i])
        # println("setting initial val of z$(i) to $(z₀[i])")
    end

    if !isnothing(λ₀)
        MOI.set(solver, MOI.NLPBlockDualStart(), λ₀)
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    λ = MOI.get(solver, MOI.NLPBlockDual())
    X = vcat([[result[idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
    U = vcat([vcat([result[idx.u[c][k]] for k in 1:Nkb], [input_idle(id) for k in 1:Nkc]) for c in 1:Nc]...)
    Δt = nothing
    if isnothing(problem.Δtb)
        Δt = [[result[idx.Δt[c][k]] for k in 1:Nkb] for c in 1:Nc]
    end

    return result, X, U, Δt, λ
end

end