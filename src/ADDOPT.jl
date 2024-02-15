module ADDOPT

using PrettyTables
using MathOptInterface, Ipopt
using LinearAlgebra, ForwardDiff
using Symbolics, SparseArrays, SparseDiffTools
using SparseArrays: findnz, nnz, nonzeros
const MOI = MathOptInterface
import HSL_jll

export AdditiveProblem, optimize_trajectory
export Furnace, FurnaceSimple
export PlanarLPBF, PlanarLPBFPrescribedMotion
export WAAMPrescribedMotion, WAAMHardnessPrescribedMotion, WAAMHardnessPrescribedTemp
export QuadraticObjective, TimeWeightedQuadraticObjective
export temperature!, combined_dynamics!
export marshall_z, initial_guess, resample_vector_traj, rollout, solve_RK4
export gen_knots, gen_fill_ref, gen_xyz, gen_torch_ref, generate_wall_z₀, row_col
export traj_to_lines, lines_to_rapid
export animate_measurement_history, animate_state_history, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar
export field_to_spots, spots_to_field, refine_grid

abstract type Dynamics end

abstract type InputDynamics <: Dynamics end
abstract type TransferDynamics <: Dynamics end
abstract type PropertyDynamics <: Dynamics end

function kc2zi(k, c, idx)
    zi = k

    for ci in 1:(c-1)
        zi += idx.Nkb[ci] + idx.Nkc[ci]
    end

    return zi
end

Nu(id::InputDynamics) = 0
Nr(id::InputDynamics) = 0
Nα(pd::PropertyDynamics) = 0
Ns(td::TransferDynamics) = 0

Nc_ineq(id::Dynamics) = 0
Nc_eq(id::Dynamics) = 0

function equality_constraint!(id::InputDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
end

function inequality_constraint!(id::InputDynamics, c::AbstractVector{Ty}, r, u, t, zi) where {Ty}
end

ineq_min(id::Dynamics) = []
ineq_max(id::Dynamics) = []

include("processes.jl")
include("objectives.jl")
include("rollout.jl")
include("initial_guess.jl")
include("visualization.jl")
include("slicer.jl")
include("transpiler.jl")
include("approximator.jl")

struct CachePackage
    fₖ_cache::Dict{Tuple{DataType,Int},Any}
    fₖ₊₁_cache::Dict{Tuple{DataType,Int},Any}
    fₘ_cache::Dict{Tuple{DataType,Int},Any}
    xₘ_cache::Dict{Tuple{DataType,Int},Any}
    ẋₘ_cache::Dict{Tuple{DataType,Int},Any}
    fₖ_cache_lock::Threads.SpinLock
    fₖ₊₁_cache_lock::Threads.SpinLock
    fₘ_cache_lock::Threads.SpinLock
    xₘ_cache_lock::Threads.SpinLock
    ẋₘ_cache_lock::Threads.SpinLock
end

struct ProblemIndex
    Nkb::Vector{Int} # Number of knots per build cycle
    Nkc::Vector{Int} # Number of knots per cooling cycle
    Nc::Int          # Number of cycles

    Nx::Int # Number of states
    Nu::Int # Number of inputs

    Neq::Int
    Nineq::Int
    Nconb::Int
    Nconc::Int
    Nconstr::Int

    Nz::Int

    u::Vector{Vector{UnitRange{Int}}}
    x::Vector{Vector{UnitRange{Int}}}
    Δtb::Vector{Vector{Int}}
    Δtc::Vector{Vector{Int}}
    constr::Vector{Vector{UnitRange{Int}}}

    free_build_time::Bool
    free_cool_time::Bool
end

struct AdditiveProblem{OB<:Objective,ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics} <: MOI.AbstractNLPEvaluator
    process::Process{ID,TD,PD}
    objective::OB

    Δtb::Union{Real,Nothing}
    Δtc::Union{Real,Nothing}

    constraint_jacobian_sparsity

    idx::ProblemIndex
    sparsity_cache

    x₀::Vector{Float64}
    ximin::Vector{Float64}
    x̄::Vector{Float64}
    xfmin::Vector{Float64}

    final_constraint::Bool
    hessian::Bool

    cp::CachePackage

    Δtb_min::Float64
    Δtb_max::Float64
    Δtc_min::Float64
    Δtc_max::Float64

    boxconstraints::Vector{Tuple{Int,Int,Vector{Float64},Vector{Float64}}}

    function AdditiveProblem(process::Process{ID,TD,PD}, objective::OB,
        Nkb, Nkc, Nc, x₀; x̄=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=true, ximin=nothing, xfmin=nothing,
        boxconstraints=nothing,
        Δtb_min=0.01,
        Δtb_max=0.04,
        Δtc_min=0.01,
        Δtc_max=0.04) where {OB<:Objective,ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics}
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        idx = generate_z_indices(
            typeof(Nkb) == Int ? Nkb * ones(Int, Nc) : Nkb,
            typeof(Nkc) == Int ? Nkc * ones(Int, Nc) : Nkc,
            Nc, Nu(id), Nr(id), Ns(td), Nα(pd), Nc_eq(id), Nc_ineq(id),
            isnothing(Δtb), isnothing(Δtc))

        fₖ_cache = Dict{Tuple{DataType,Int},Any}()
        fₖ₊₁_cache = Dict{Tuple{DataType,Int},Any}()
        fₘ_cache = Dict{Tuple{DataType,Int},Any}()
        xₘ_cache = Dict{Tuple{DataType,Int},Any}()
        ẋₘ_cache = Dict{Tuple{DataType,Int},Any}()
        fₖ_cache_lock = Threads.SpinLock()
        fₖ₊₁_cache_lock = Threads.SpinLock()
        fₘ_cache_lock = Threads.SpinLock()
        xₘ_cache_lock = Threads.SpinLock()
        ẋₘ_cache_lock = Threads.SpinLock()
        cp = CachePackage(fₖ_cache, fₖ₊₁_cache, fₘ_cache, xₘ_cache, ẋₘ_cache, fₖ_cache_lock, fₖ₊₁_cache_lock, fₘ_cache_lock, xₘ_cache_lock, ẋₘ_cache_lock)

        println("Preparing sparsity")

        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, process, cp; Δtb=Δtb, Δtc=Δtc)
        println("Done with sparsity")

        if isnothing(ximin)
            ximin = x₀
        end

        if !final_constraint
            xfmin = -Inf * ones(idx.Nx)
            x̄ = Inf * ones(idx.Nx)
        elseif isnothing(xfmin)
            xfmin = x̄
        end

        if isnothing(boxconstraints)
            boxconstraints = []
        end

        new{OB,ID,TD,PD}(process, objective, Δtb, Δtc, con_jacobian_sparsity,
            idx, sparsity_cache,
            x₀, ximin, x̄, xfmin,
            final_constraint, hessian,
            cp,
            Δtb_min, Δtb_max,
            Δtc_min, Δtc_max,
            boxconstraints)
    end
end

function generate_z_indices(Nkb, Nkc, Nc, Nu, Nr, Ns, Nα, Neq, Nineq, free_build_time, free_cool_time)
    Nx = Ns + Nα + Nr
    NΔtb = free_build_time ? 1 : 0
    NΔtc = free_cool_time ? 1 : 0
    Nconb = Nx + Neq + Nineq
    Nconc = Nx

    Nconstr = 0
    Nz = 0

    u::Vector{Vector{UnitRange{Int}}} = []
    x::Vector{Vector{UnitRange{Int}}} = []
    Δtb::Vector{Vector{Int}} = []
    Δtc::Vector{Vector{Int}} = []
    constr::Vector{Vector{UnitRange{Int}}} = []

    i = 1
    for c in 1:Nc
        u_cyc::Vector{UnitRange{Int}} = []
        x_cyc::Vector{UnitRange{Int}} = []
        Δtb_cyc::Vector{Int} = []
        Δtc_cyc::Vector{Int} = []
        constr_cyc::Vector{UnitRange{Int}} = []

        for k in 1:Nkb[c] # build step
            push!(x_cyc, i:(i-1+Nx))
            i += Nx
            push!(u_cyc, i:(i-1+Nu))
            i += Nu
            if free_build_time
                push!(Δtb_cyc, i)
                i += NΔtb
            end

            Nz += Nx + Nu + NΔtb

            push!(constr_cyc, (1+Nconstr):(Nconstr+Nconb))
            Nconstr += Nconb
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c]) # cooling step
            push!(x_cyc, i:(i-1+Nx))
            i += Nx
            if free_cool_time
                push!(Δtc_cyc, i)
                i += NΔtc
            end

            Nz += Nx + NΔtc

            if c < Nc || k < Nkb[c] + Nkc[c]
                push!(constr_cyc, (1+Nconstr):(Nconstr+Nconc))
                Nconstr += Nconc
            end
        end

        push!(x, x_cyc)
        push!(u, u_cyc)
        push!(Δtb, Δtb_cyc)
        push!(Δtc, Δtc_cyc)
        push!(constr, constr_cyc)
    end

    return ProblemIndex(
        Nkb, Nkc, Nc,
        Nx, Nu,
        Neq, Nineq, Nconb, Nconc,
        Nconstr, Nz,
        u, x, Δtb, Δtc, constr,
        free_build_time, free_cool_time)
end

function combined_dynamics!(f, x, u, process::Process{ID,TD,PD}, t, zi, Δt) where {ID,TD,PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics

    s = view(x, 1:Ns(td))
    α = view(x, (Ns(td)+1):(Ns(td)+Nα(pd)))
    r = view(x, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    ds = view(f, 1:Ns(td))
    dα = view(f, (Ns(td)+1):(Ns(td)+Nα(pd)))
    dr = view(f, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    dynamics_function!(td, ds, s, t, zi)
    input_function!(id, ds, r, u, t, zi, Δt) # always call second, additive
    dynamics_function!(pd, td, dα, α, s, t, zi)
    dynamics_function!(id, dr, s, r, u, t, zi)
end

function collocation_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    Nx = length(xₖ)

    fₖ_cache = cp.fₖ_cache
    fₖ₊₁_cache = cp.fₖ₊₁_cache
    fₘ_cache = cp.fₘ_cache
    xₘ_cache = cp.xₘ_cache
    ẋₘ_cache = cp.ẋₘ_cache

    thread::Int = Threads.threadid()
    fₖ = get!(fₖ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}

    fₖ₊₁ = get!(fₖ₊₁_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}

    fₘ = get!(fₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}

    xₘ = get!(xₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}

    ẋₘ = get!(ẋₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}

    combined_dynamics!(fₖ, xₖ, uₖ, process, t, zi, Δt[1])
    combined_dynamics!(fₖ₊₁, xₖ₊₁, uₖ, process, t + Δt[1], zi + 1, Δt[1])

    xₘ .= @. 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    ẋₘ .= @. (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    combined_dynamics!(fₘ, xₘ, uₖ, process, t + Δt[1] / 2.0, zi + 0.5, Δt[1])
    r .= fₘ .- ẋₘ
end

function equality_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, t, zi) where {T,ID,TD,PD}
    ns = Ns(process.transfer_dynamics)
    nα = Nα(process.property_dynamics)
    nr = Nr(process.input_dynamics)
    rₖ = view(xₖ, (ns+nα+1):(ns+nα+nr))

    equality_constraint!(process.input_dynamics, r, rₖ, uₖ, t, zi)
end

function inequality_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, t, zi) where {T,ID,TD,PD}
    ns = Ns(process.transfer_dynamics)
    nα = Nα(process.property_dynamics)
    nr = Nr(process.input_dynamics)
    rₖ = view(xₖ, (ns+nα+1):(ns+nα+nr))

    inequality_constraint!(process.input_dynamics, r, rₖ, uₖ, t, zi)
end

function total_build_constraint!(process::Process{ID,TD,PD}, idx::ProblemIndex, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    Nx, Neq, Nineq = idx.Nx, idx.Neq, idx.Nineq
    r_colloc = view(r, 1:Nx)
    r_eq = view(r, (Nx+1):(Nx+Neq))
    r_ineq = view(r, (Nx+Neq+1):(Nx+Neq+Nineq))

    collocation_constraint!(process, r_colloc, xₖ, uₖ, xₖ₊₁, Δt, t, zi, cp)
    equality_constraint!(process, r_eq, xₖ, uₖ, t, zi)
    inequality_constraint!(process, r_ineq, xₖ, uₖ, t, zi)
end

function total_cooling_constraint!(process::Process{ID,TD,PD}, idx::ProblemIndex, r::AbstractVector{T}, xₖ, xₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    ui = input_idle(process.input_dynamics)

    collocation_constraint!(process, r, xₖ, ui, xₖ₊₁, Δt, t, zi, cp)
end

function constraints!(process::Process, c, z, idx::ProblemIndex, cp::CachePackage; xf=nothing, Δtb=nothing, Δtc=nothing)
    # Collocation, Initial state, Final State
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)
    Nconb, Nconc = idx.Nconb, idx.Nconc
    constr = idx.constr

    t = 0.0
    for cyc in 1:Nc
        if cyc > 1
            if free_cool_time
                Δtc = z[idx.Δtc[cyc-1][end]]
            end
            zi = kc2zi(Nkb[cyc] + Nkc[cyc], cyc - 1, idx)

            xₖ = @view z[idx.x[cyc-1][end]]
            xₖ₊₁ = @view z[idx.x[cyc][1]]
            r = @view c[constr[cyc-1][end]]

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t, zi, cp)
            # t += Δtc # TEMPORARY
        end

        for k in 1:(Nkb[cyc]-1)
            if free_build_time
                Δtb = z[idx.Δtb[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            r = @view c[constr[cyc][k]]

            total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi, cp)
            # t += Δtb # TEMPORARY
        end

        k = Nkb[cyc]
        if free_build_time
            Δtb = z[idx.Δtb[cyc][k]]
        end
        zi = kc2zi(k, cyc, idx)

        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)
        r = @view c[constr[cyc][k]]

        total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb, t, zi, cp)
        # t += Δtb # TEMPORARY

        for k in (Nkb[cyc]+1):(Nkb[cyc]+Nkc[cyc]-1)
            if free_cool_time
                Δtc = z[idx.Δtc[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            r = @view c[constr[cyc][k]]

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t, zi, cp)
            # t += Δtc # TEMPORARY
        end

    end
end

function constraint_jacobian!(process::Process, jac, z, idx::ProblemIndex, sparsity_cache, cp::CachePackage; Δtb=nothing, Δtc=nothing, final_constraint=false, prob=nothing)
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc

    conb_∂xₖ_sparsity, conb_∂uₖ_sparsity, conb_∂xₖ₊₁_sparsity, conb_∂uₖ₊₁_sparsity, conb_∂Δt_sparsity, conc_∂xₖ_sparsity, conc_∂xₖ₊₁_sparsity, conc_∂Δt_sparsity, conb_∂uₖ_color, conb_∂xₖ_color, conb_∂xₖ₊₁_color, conb_∂uₖ₊₁_color, conb_∂Δt_color, conc_∂xₖ_color, conc_∂xₖ₊₁_color, conc_∂Δt_color, jac_cache_conb_∂Δt, jac_cache_conb_∂xₖ, jac_cache_conb_∂uₖ, jac_cache_conb_∂xₖ₊₁, jac_cache_conb_∂uₖ₊₁, jac_cache_conc_∂Δt, jac_cache_conc_∂xₖ, jac_cache_conc_∂xₖ₊₁ = sparsity_cache
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    conb_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X, t, zi, cp), Δt, jac_cache_conb_∂Δt)
    conb_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp), xₖ, jac_cache_conb_∂xₖ)
    conb_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp), uₖ, jac_cache_conb_∂uₖ)
    conb_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, X, uₖ₊₁, Δt, t, zi, cp), xₖ₊₁, jac_cache_conb_∂xₖ₊₁)
    conb_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, X, Δt, t, zi, cp), uₖ₊₁, jac_cache_conb_∂uₖ₊₁)

    conc_∂Δt(J, xₖ, xₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, X, t, zi, cp), Δt, jac_cache_conc_∂Δt)
    conc_∂xₖ(J, xₖ, xₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, X, xₖ₊₁, Δt, t, zi, cp), xₖ, jac_cache_conc_∂xₖ)
    conc_∂xₖ₊₁(J, xₖ, xₖ₊₁, Δt, t, zi) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, xₖ, X, Δt, t, zi, cp), xₖ₊₁, jac_cache_conc_∂xₖ₊₁)

    Jb∂xₖ = Float64.(sparse(conb_∂xₖ_sparsity))
    Jb∂uₖ = Float64.(sparse(conb_∂uₖ_sparsity))
    Jb∂xₖ₊₁ = Float64.(sparse(conb_∂xₖ₊₁_sparsity))
    Jb∂uₖ₊₁ = Float64.(sparse(conb_∂uₖ₊₁_sparsity))
    Jb∂Δt = Float64.(sparse(conb_∂Δt_sparsity))

    lJb∂xₖ = nnz(Jb∂xₖ)
    lJb∂uₖ = nnz(Jb∂uₖ)
    lJb∂xₖ₊₁ = nnz(Jb∂xₖ₊₁)
    lJb∂uₖ₊₁ = nnz(Jb∂uₖ₊₁)
    lJb∂Δt = free_build_time ? nnz(Jb∂Δt) : 0

    Jc∂xₖ = Float64.(sparse(conc_∂xₖ_sparsity))
    Jc∂xₖ₊₁ = Float64.(sparse(conc_∂xₖ₊₁_sparsity))
    Jc∂Δt = Float64.(sparse(conc_∂Δt_sparsity))

    lJc∂xₖ = nnz(Jc∂xₖ)
    lJc∂xₖ₊₁ = nnz(Jc∂xₖ₊₁)
    lJc∂Δt = free_cool_time ? nnz(Jc∂Δt) : 0

    Nconbjac = lJb∂xₖ + lJb∂uₖ + lJb∂xₖ₊₁ + lJb∂uₖ₊₁ + lJb∂Δt
    Nconcjac = lJc∂xₖ + lJc∂xₖ₊₁ + lJc∂Δt

    t = 0.0
    i = 1
    for cyc in 1:Nc

        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][end]]
            end
            zi = kc2zi(Nkb[cyc] + Nkc[cyc], cyc - 1, idx)

            xₖ = @view z[idx.x[cyc-1][end]]
            xₖ₊₁ = @view z[idx.x[cyc][1]]

            conc_∂xₖ(Jc∂xₖ, xₖ, xₖ₊₁, Δtc, t, zi)
            view(jac, i:(i+lJc∂xₖ-1)) .= nonzeros(Jc∂xₖ)
            i += lJc∂xₖ

            if free_cool_time
                conc_∂Δt(Jc∂Δt, xₖ, xₖ₊₁, Δtc, t, zi)
                view(jac, i:(i+lJc∂Δt-1)) .= nonzeros(Jc∂Δt)
                i += lJc∂Δt
            end

            conc_∂xₖ₊₁(Jc∂xₖ₊₁, xₖ, xₖ₊₁, Δtc, t, zi)
            view(jac, i:(i+lJc∂xₖ₊₁-1)) .= nonzeros(Jc∂xₖ₊₁)
            i += lJc∂xₖ₊₁
            # t += Δtc[1] # TEMPORARY
        end

        for k in 1:(Nkb[cyc]-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]

            conb_∂xₖ(Jb∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
            view(jac, i:(i+lJb∂xₖ-1)) .= nonzeros(Jb∂xₖ)
            i += lJb∂xₖ

            conb_∂uₖ(Jb∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
            view(jac, i:(i+lJb∂uₖ-1)) .= nonzeros(Jb∂uₖ)
            i += lJb∂uₖ

            if free_build_time
                conb_∂Δt(Jb∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
                view(jac, i:(i+lJb∂Δt-1)) .= nonzeros(Jb∂Δt)
                i += lJb∂Δt
            end

            conb_∂xₖ₊₁(Jb∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
            view(jac, i:(i+lJb∂xₖ₊₁-1)) .= nonzeros(Jb∂xₖ₊₁)
            i += lJb∂xₖ₊₁

            conb_∂uₖ₊₁(Jb∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
            view(jac, i:(i+lJb∂uₖ₊₁-1)) .= nonzeros(Jb∂uₖ₊₁)
            i += lJb∂uₖ₊₁

            # t += Δtb[1] # TEMPORARY
        end

        k = Nkb[cyc]
        if free_build_time
            Δtb = @view z[idx.Δtb[cyc][k]]
        end
        zi = kc2zi(k, cyc, idx)

        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uₖ₊₁ = input_idle(process.input_dynamics)

        conb_∂xₖ(Jb∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
        view(jac, i:(i+lJb∂xₖ-1)) .= nonzeros(Jb∂xₖ)
        i += lJb∂xₖ

        conb_∂uₖ(Jb∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
        view(jac, i:(i+lJb∂uₖ-1)) .= nonzeros(Jb∂uₖ)
        i += lJb∂uₖ

        if free_build_time
            conb_∂Δt(Jb∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
            view(jac, i:(i+lJb∂Δt-1)) .= nonzeros(Jb∂Δt)
            i += lJb∂Δt
        end

        conb_∂xₖ₊₁(Jb∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi)
        view(jac, i:(i+lJb∂xₖ₊₁-1)) .= nonzeros(Jb∂xₖ₊₁)
        i += lJb∂xₖ₊₁
        # t += Δtb[1] # TEMPORARY

        for k in (Nkb[cyc]+1):(Nkb[cyc]+Nkc[cyc]-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]

            conc_∂xₖ(Jc∂xₖ, xₖ, xₖ₊₁, Δtc, t, zi)
            view(jac, i:(i+lJc∂xₖ-1)) .= nonzeros(Jc∂xₖ)
            i += lJc∂xₖ

            if free_cool_time
                conc_∂Δt(Jc∂Δt, xₖ, xₖ₊₁, Δtc, t, zi)
                view(jac, i:(i+lJc∂Δt-1)) .= nonzeros(Jc∂Δt)
                i += lJc∂Δt
            end

            conc_∂xₖ₊₁(Jc∂xₖ₊₁, xₖ, xₖ₊₁, Δtc, t, zi)
            view(jac, i:(i+lJc∂xₖ₊₁-1)) .= nonzeros(Jc∂xₖ₊₁)
            i += lJc∂xₖ₊₁
            # t += Δtc[1] # TEMPORARY
        end
    end

    # res = zeros(idx.Nconstr, idx.Nz)
    # rp = zeros(idx.Nconstr)
    # ForwardDiff.jacobian!(res, (r, z) -> constraints!(process, r, z, idx, cp; xf=prob.x̄, Δtb=Δtb, Δtc=Δtc), rp, z)
    # # println("reference")#tf = borderless,
    # # pretty_table(res,  noheader = true, crop = :none, formatters = ft_printf("%3.1e"))
    # # display(res)
    # # show(stdout, "text/plain", res)
    # # display(sparse(res))

    # rs = [r for (r, c) in prob.constraint_jacobian_sparsity]
    # cs = [c for (r, c) in prob.constraint_jacobian_sparsity]
    # # println("actual")
    # # pretty_table(sparse(rs, cs, jac), noheader = true, crop = :none, formatters = ft_printf("%3.1e"))
    # # display(Matrix(sparse(rs, cs, jac)))
    # # show(stdout, "text/plain", Matrix(sparse(rs, cs, jac)))
    # display(norm(sparse(rs, cs, jac)-sparse(res)))
end

function constraint_jacobian_sparsity(idx::ProblemIndex, process::Process, cp::CachePackage; Δtb=nothing, Δtc=nothing)
    Nx, Nu = idx.Nx, idx.Nu
    Nkb, Nkc, Nc = idx.Nkb, idx.Nkc, idx.Nc
    Nconb, Nconc = idx.Nconb, idx.Nconc
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    rdb = ones(Symbolics.Num, Nconb)
    rdc = ones(Symbolics.Num, Nconc)
    xₖ = 0.002 * ones(Symbolics.Num, Nx)
    uₖ = 0.003 * ones(Symbolics.Num, Nu)
    xₖ₊₁ = 0.004 * ones(Symbolics.Num, Nx)
    uₖ₊₁ = 0.005 * ones(Symbolics.Num, Nu)
    Δtd = 0.006 * ones(Symbolics.Num, 1)
    t = 0.0
    zi = 1

    conb_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δtd, t, zi, cp), rdb, xₖ)
    conb_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δtd, t, zi, cp), rdb, uₖ)
    conb_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, X, uₖ₊₁, Δtd, t, zi, cp), rdb, xₖ₊₁)
    conb_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, X, Δtd, t, zi, cp), rdb, uₖ₊₁)
    conb_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X, t, zi, cp), rdb, Δtd)

    conc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, X, xₖ₊₁, Δtd, t, zi, cp), rdc, xₖ)
    conc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, X, Δtd, t, zi, cp), rdc, xₖ₊₁)
    conc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, X, t, zi, cp), rdc, Δtd)

    println("Computing sparsity xₖ")
    display(conb_∂xₖ_sparsity)
    println("Computing sparsity uₖ")
    display(conb_∂uₖ_sparsity)

    if free_build_time || free_cool_time
        display(conb_∂Δt_sparsity)
    end

    conb_∂uₖ_color = matrix_colors(Float64.(conb_∂uₖ_sparsity))
    conb_∂xₖ_color = matrix_colors(Float64.(conb_∂xₖ_sparsity))
    conb_∂xₖ₊₁_color = matrix_colors(Float64.(conb_∂xₖ₊₁_sparsity))
    conb_∂uₖ₊₁_color = matrix_colors(Float64.(conb_∂uₖ₊₁_sparsity))
    conb_∂Δt_color = matrix_colors(Float64.(conb_∂Δt_sparsity))

    conc_∂xₖ_color = matrix_colors(Float64.(conc_∂xₖ_sparsity))
    conc_∂xₖ₊₁_color = matrix_colors(Float64.(conc_∂xₖ₊₁_sparsity))
    conc_∂Δt_color = matrix_colors(Float64.(conc_∂Δt_sparsity))

    dxb = zeros(Nconb)
    dxc = zeros(Nconc)
    xₖ = randn(Nx)
    uₖ = randn(Nu)
    xₖ₊₁ = randn(Nx)
    uₖ₊₁ = randn(Nu)
    Δt = randn(1)
    t = 0.0
    zi = 1

    jac_cache_conb_∂Δt = ForwardColorJacCache((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X, t, zi, cp), Δt, nothing; dx=dxb, colorvec=conb_∂Δt_color, sparsity=conb_∂Δt_sparsity)
    jac_cache_conb_∂xₖ = ForwardColorJacCache((r, X) -> total_build_constraint!(process, idx, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp), xₖ, nothing; dx=dxb, colorvec=conb_∂xₖ_color, sparsity=conb_∂xₖ_sparsity)
    jac_cache_conb_∂uₖ = ForwardColorJacCache((r, X) -> total_build_constraint!(process, idx, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp), uₖ, nothing; dx=dxb, colorvec=conb_∂uₖ_color, sparsity=conb_∂uₖ_sparsity)
    jac_cache_conb_∂xₖ₊₁ = ForwardColorJacCache((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, X, uₖ₊₁, Δt, t, zi, cp), xₖ₊₁, nothing; dx=dxb, colorvec=conb_∂xₖ₊₁_color, sparsity=conb_∂xₖ₊₁_sparsity)
    jac_cache_conb_∂uₖ₊₁ = ForwardColorJacCache((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, X, Δt, t, zi, cp), uₖ₊₁, nothing; dx=dxb, colorvec=conb_∂uₖ₊₁_color, sparsity=conb_∂uₖ₊₁_sparsity)
    jac_cache_conc_∂Δt = ForwardColorJacCache((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, X, t, zi, cp), Δt, nothing; dx=dxc, colorvec=conc_∂Δt_color, sparsity=conc_∂Δt_sparsity)
    jac_cache_conc_∂xₖ = ForwardColorJacCache((r, X) -> total_cooling_constraint!(process, idx, r, X, xₖ₊₁, Δt, t, zi, cp), xₖ, nothing; dx=dxc, colorvec=conc_∂xₖ_color, sparsity=conc_∂xₖ_sparsity)
    jac_cache_conc_∂xₖ₊₁ = ForwardColorJacCache((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, X, Δt, t, zi, cp), xₖ₊₁, nothing; dx=dxc, colorvec=conc_∂xₖ₊₁_color, sparsity=conc_∂xₖ₊₁_sparsity)

    total_structure = Vector{Tuple{Int,Int}}()
    row_offset = 0

    println("Entering loop")
    for cyc in 1:Nc
        if cyc > 1
            r, c, _ = findnz(conc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc-1][end][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(conc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtc[cyc-1][end][1] .- 1)))
            end

            r, c, _ = findnz(conc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][1][1] .- 1)))

            row_offset += Nconc
        end

        for k in 1:(Nkb[cyc]-1)
            r, c, _ = findnz(conb_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

            r, c, _ = findnz(conb_∂uₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.u[cyc][k][1] .- 1)))

            if free_build_time
                r, c, _ = findnz(conb_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc][k][1] .- 1)))
            end

            r, c, _ = findnz(conb_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

            r, c, _ = findnz(conb_∂uₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.u[cyc][k+1][1] .- 1)))

            row_offset += Nconb
        end

        k = Nkb[cyc]
        r, c, _ = findnz(conb_∂xₖ_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

        r, c, _ = findnz(conb_∂uₖ_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.u[cyc][k][1] .- 1)))

        if free_build_time
            r, c, _ = findnz(conb_∂Δt_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc][k][1] .- 1)))
        end

        r, c, _ = findnz(conb_∂xₖ₊₁_sparsity)
        append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

        row_offset += Nconb

        for k in (Nkb[cyc]+1):(Nkb[cyc]+Nkc[cyc]-1)
            r, c, _ = findnz(conc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(conc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtc[cyc][k][1] .- 1)))
            end

            r, c, _ = findnz(conc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][k+1][1] .- 1)))

            row_offset += Nconc
        end

    end
    println("Finished loop")

    sparsity_cache = conb_∂xₖ_sparsity, conb_∂uₖ_sparsity, conb_∂xₖ₊₁_sparsity, conb_∂uₖ₊₁_sparsity, conb_∂Δt_sparsity, conc_∂xₖ_sparsity, conc_∂xₖ₊₁_sparsity, conc_∂Δt_sparsity, conb_∂uₖ_color, conb_∂xₖ_color, conb_∂xₖ₊₁_color, conb_∂uₖ₊₁_color, conb_∂Δt_color, conc_∂xₖ_color, conc_∂xₖ₊₁_color, conc_∂Δt_color, jac_cache_conb_∂Δt, jac_cache_conb_∂xₖ, jac_cache_conb_∂uₖ, jac_cache_conb_∂xₖ₊₁, jac_cache_conb_∂uₖ₊₁, jac_cache_conc_∂Δt, jac_cache_conc_∂xₖ, jac_cache_conc_∂xₖ₊₁
    return total_structure, sparsity_cache
end

function constraint_hessian_structure(prob::AdditiveProblem)
    structure = []
    idx = prob.idx

    for cyc in 1:Nc
        if cyc > 1
            xₖ = idx.x[cyc-1][end]
            Δt = idx.Δtc[cyc-1][end]
            xₖ₊₁ = idx.x[cyc][1]
            zₖ = xₖ[1]:xₖ₊₁[end]
        end

        for k in 1:(Nkb[cyc]-1)
            xₖ = idx.x[cyc][k]
            uₖ = idx.u[cyc][k]
            Δt = idx.Δtb[cyc][k]
            xₖ₊₁ = idx.x[cyc][k+1]
            uₖ₊₁ = idx.u[cyc][k+1]
            zₖ = xₖ[1]:uₖ₊₁[end]
        end

        k = Nkb[cyc]
        xₖ = idx.x[cyc][k]
        uₖ = idx.u[cyc][k]
        Δt = idx.Δtb[cyc][k]
        xₖ₊₁ = idx.x[cyc][k+1]
        uₖ₊₁ = idx.u[cyc][k+1]
        zₖ = xₖ[1]:uₖ₊₁[end]

        for k in (Nkb[cyc]+1):(Nkb[cyc]+Nkc[cyc]-1)
            xₖ = idx.x[cyc][k]
            Δt = idx.Δtc[cyc][k]
            xₖ₊₁ = idx.x[cyc][k+1]
            zₖ = xₖ[1]:xₖ₊₁[end]
        end
    end

    return structure
end

function constraint_hessian_values(prob::AdditiveProblem, H, z, μ)
    idx = prob.idx

    for cyc in 1:Nc
        if cyc > 1
            xₖ = idx.x[cyc-1][end]
            Δt = idx.Δtc[cyc-1][end]
            xₖ₊₁ = idx.x[cyc][1]
            zₖ = xₖ[1]:xₖ₊₁[end]
            # ForwardDiff.jacobian!()
            (r, z) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtk, t, zi, cp)
        end

        for k in 1:(Nkb[cyc]-1)
            # (r, z) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtd, t, zi, cp)
            xₖ = idx.x[cyc][k]
            uₖ = idx.u[cyc][k]
            Δt = idx.Δtb[cyc][k]
            xₖ₊₁ = idx.x[cyc][k+1]
            uₖ₊₁ = idx.u[cyc][k+1]
            zₖ = xₖ[1]:uₖ₊₁[end]
        end

        k = Nkb[cyc]
        xₖ = idx.x[cyc][k]
        uₖ = idx.u[cyc][k]
        Δt = idx.Δtb[cyc][k]
        xₖ₊₁ = idx.x[cyc][k+1]
        uₖ₊₁ = idx.u[cyc][k+1]
        zₖ = xₖ[1]:uₖ₊₁[end]

        for k in (Nkb[cyc]+1):(Nkb[cyc]+Nkc[cyc]-1)
            xₖ = idx.x[cyc][k]
            Δt = idx.Δtc[cyc][k]
            xₖ₊₁ = idx.x[cyc][k+1]
            zₖ = xₖ[1]:xₖ₊₁[end]
        end
    end
end

function MOI.eval_objective(prob::AdditiveProblem, z)
    return cost(prob.objective, z, prob.idx)
end

function MOI.eval_objective_gradient(prob::AdditiveProblem, grad_f, z)
    gradient(prob.objective, grad_f, z, prob.idx)
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    constraints!(prob.process, c, z, prob.idx, prob.cp, xf=prob.x̄, Δtb=prob.Δtb, Δtc=prob.Δtc)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.sparsity_cache, prob.cp, Δtb=prob.Δtb, Δtc=prob.Δtc, final_constraint=prob.final_constraint, prob=prob)
end

function MOI.hessian_lagrangian_structure(prob::AdditiveProblem)
    structure = objective_hessian_structure(prob.objective, prob.idx)
    # append!(structure, constraint_hessian_structure(prob))

    return structure
end

function MOI.eval_hessian_lagrangian(prob::AdditiveProblem, H, z, σ, μ)
    objective_hessian_values(prob.objective, prob.idx, H, z)
    H .*= σ

    # ### TEMP EXPERIMENT
    # i = 0
    # for c in 1:prob.idx.Nc
    #     for k in 1:prob.idx.Nkb[c]
    #         i += prob.idx.Nx
    #         H[(i+1):(i+prob.idx.Nu)] .= μ[406 + (k-1)*406]
    #         i += prob.idx.Nu
    #     end
    # end

    # constraint_hessian_values(prob, H, z, μ)
end

function MOI.features_available(prob::AdditiveProblem)
    if prob.hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end

MOI.initialize(prob::AdditiveProblem, features) = nothing
MOI.jacobian_structure(prob::AdditiveProblem) = prob.constraint_jacobian_sparsity

function optimize_trajectory(problem::AdditiveProblem;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing, λ₀=nothing, ug=nothing, xg=nothing, solv="ma97", isqp=false)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["hsllib"] = HSL_jll.libhsl_path
    solver.options["linear_solver"] = solv

    if isqp
        solver.options["hessian_constant"] = "yes"
        solver.options["jac_c_constant"] = "yes"
        solver.options["jac_d_constant"] = "yes"
    end

    idx = problem.idx
    Nz, Nconstr = idx.Nz, idx.Nconstr
    Nx, Neq, Nineq = idx.Nx, idx.Neq, idx.Nineq
    process = problem.process
    Nkb, Nkc, Nc = idx.Nkb, idx.Nkc, idx.Nc
    id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics

    if isnothing(z₀)
        z₀ = zeros(Nz)
        for c in 1:Nc

            for k in 1:Nkb[c]
                if isnothing(problem.Δtb)
                    z₀[idx.Δtb[c][k]] = (problem.Δtb_min + problem.Δtb_max) / 2
                end
                z₀[idx.x[c][k]] .= isnothing(xg) ? problem.x̄ : xg
                z₀[idx.u[c][k]] .= isnothing(ug) ? input_min(id) : ug
            end

            for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
                if isnothing(problem.Δtc)
                    z₀[idx.Δtc[c][k]] = (problem.Δtc_min + problem.Δtc_max) / 2
                end
                z₀[idx.x[c][k]] .= isnothing(xg) ? problem.x̄ : xg
            end
        end
    end
    @show norm(z₀)
    @show isnothing(problem.Δtb)
    @show isnothing(problem.Δtc)

    ct = zeros(Nconstr)
    gt = zeros(Nz)
    gr = zeros(Nz)
    jt = zeros(length(problem.constraint_jacobian_sparsity))
    μ0 = ones(Nconstr)

    @show Nz
    println("Checking objective function...")
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @show MOI.eval_objective(problem, z₀)
    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, z₀)
    @time MOI.eval_objective_gradient(problem, gt, z₀)

    if problem.hessian
        println("Checking lagrangian hessian...")
        structure = MOI.hessian_lagrangian_structure(problem)
        H0 = zeros(length(structure))

        rs = [r for (r, c) in structure]
        cs = [c for (r, c) in structure]
        @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
        @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
    end

    println("Checking constraint function...")
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @show maximum(abs.(ct))
    println("Checking constraint jacobian...")
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)

    c_lb = [zeros(Nx + Neq); ineq_min(id)]
    c_ub = [zeros(Nx + Neq); ineq_max(id)]
    c_lc = zeros(Nx)
    c_uc = zeros(Nx)

    c_l = []
    c_u = []
    for c in 1:Nc
        for k in 1:Nkb[c]
            append!(c_l, c_lb)
            append!(c_u, c_ub)
        end
        for k in 1:Nkc[c]
            if c < Nc || k < Nkc[c]
                append!(c_l, c_lc)
                append!(c_u, c_uc)
            end
        end
    end

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, problem, true)

    z = MOI.add_variables(solver, Nz)
    x_min = [state_min(td)
        property_min(pd)
        state_min(id)]
    x_max = [state_max(td)
        property_max(pd)
        state_max(id)]

    x_l = [Vector{Any}(undef, Nkb[c] + Nkc[c]) for c in 1:Nc]
    x_u = [Vector{Any}(undef, Nkb[c] + Nkc[c]) for c in 1:Nc]

    # Set primal bounds and initial values
    for c in 1:Nc
        for k in 1:Nkb[c]
            if isnothing(problem.Δtb)
                Δtb = z[idx.Δtb[c][k]]
                MOI.add_constraint(solver, Δtb, MOI.LessThan(problem.Δtb_max))
                MOI.add_constraint(solver, Δtb, MOI.GreaterThan(problem.Δtb_min))
            end

            uj = z[idx.u[c][k]]
            MOI.add_constraints(solver, uj, MOI.LessThan.(input_max(id)))
            MOI.add_constraints(solver, uj, MOI.GreaterThan.(input_min(id)))

            xj = z[idx.x[c][k]]
            if c > 1 || k > 1
                x_u[c][k] = MOI.add_constraints(solver, xj, MOI.LessThan.(x_max))
                x_l[c][k] = MOI.add_constraints(solver, xj, MOI.GreaterThan.(x_min))
            else
                # Initial state constraint
                x_u[c][k] = MOI.add_constraints(solver, xj, MOI.LessThan.(problem.x₀))
                x_l[c][k] = MOI.add_constraints(solver, xj, MOI.GreaterThan.(problem.ximin))
            end
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            if isnothing(problem.Δtc)
                Δtc = z[idx.Δtc[c][k]]

                if c == Nc # Fix size of final cooling to max
                    MOI.add_constraint(solver, Δtc, MOI.LessThan(problem.Δtc_max))
                    MOI.add_constraint(solver, Δtc, MOI.GreaterThan(problem.Δtc_max))
                else
                    MOI.add_constraint(solver, Δtc, MOI.LessThan(problem.Δtc_max))
                    MOI.add_constraint(solver, Δtc, MOI.GreaterThan(problem.Δtc_min))
                end
            end

            xj = z[idx.x[c][k]]
            if c < Nc || k < Nkb[c] + Nkc[c]
                x_u[c][k] = MOI.add_constraints(solver, xj, MOI.LessThan.(x_max))
                x_l[c][k] = MOI.add_constraints(solver, xj, MOI.GreaterThan.(x_min))
            else
                # Final state constraint
                x_u[c][k] = MOI.add_constraints(solver, xj, MOI.LessThan.(problem.x̄))
                x_l[c][k] = MOI.add_constraints(solver, xj, MOI.GreaterThan.(problem.xfmin))
            end
        end
    end

    for boxcon in problem.boxconstraints
        c, k, x_min, x_max = boxcon
        xj = z[idx.x[c][k]]

        MOI.delete(solver, x_u[c][k])
        MOI.delete(solver, x_l[c][k])
        x_l[c][k] = MOI.add_constraints(solver, xj, MOI.LessThan.(x_max))
        x_l[c][k] = MOI.add_constraints(solver, xj, MOI.GreaterThan.(x_min))
    end

    for i in 1:lastindex(z₀)
        MOI.set(solver, MOI.VariablePrimalStart(), z[i], z₀[i])
    end

    if !isnothing(λ₀)
        MOI.set(solver, MOI.NLPBlockDualStart(), λ₀)
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    flush(stdout)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    λ = MOI.get(solver, MOI.NLPBlockDual())
    X = vcat([[result[idx.x[c][k]] for k in 1:(Nkb[c]+Nkc[c])] for c in 1:Nc]...)
    U = vcat([vcat([result[idx.u[c][k]] for k in 1:Nkb[c]], [input_idle(id) for k in 1:Nkc[c]]) for c in 1:Nc]...)

    Δt = vcat([[isnothing(problem.Δtb) ? [result[idx.Δtb[c][k]] for k in 1:Nkb[c]] : problem.Δtb * ones(Nkb[c])
        isnothing(problem.Δtc) ? [result[idx.Δtc[c][k]] for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])] : problem.Δtc * ones(Nkc[c])
    ] for c in 1:Nc]...)

    return result, X, U, Δt, λ
end

end