module ADDOPT

@time using PrettyTables
@time using MathOptInterface, Ipopt
@time using LinearAlgebra, ForwardDiff
@time using Symbolics, SparseArrays, SparseDiffTools
@time using SparseArrays: findnz, nnz, nonzeros
const MOI = MathOptInterface
import HSL_jll

abstract type Dynamics end

abstract type InputDynamics <: Dynamics end
abstract type TransferDynamics <: Dynamics end
abstract type PropertyDynamics <: Dynamics end

kc2zi(k, c, idx) = k + (c - 1) * idx.Nc

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
include("goal_optimizer.jl")
include("transpiler.jl")

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
    ximin::Vector{Float64}
    x̄::Vector{Float64}

    final_constraint::Bool
    hessian::Bool

    cp::CachePackage

    Δtb_min::Float64
    Δtb_max::Float64
    Δtc_min::Float64
    Δtc_max::Float64

    function AdditiveProblem(process::Process{ID,TD,PD}, objective::OB,
        Nkb, Nkc, Nc, x₀; x̄=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=true, ximin=nothing,
        Δtb_min=0.01,
        Δtb_max=0.04,
        Δtc_min=0.01,
        Δtc_max=0.04) where {OB<:Objective,ID<:InputDynamics,TD<:TransferDynamics,PD<:PropertyDynamics}
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        idx = generate_z_indices(Nkb, Nkc, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), Nc_eq(id), Nc_ineq(id), Δtb, Δtc, final_constraint)

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

        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, process, cp; Δtb=Δtb, Δtc=Δtc, final_constraint=final_constraint)
        println("Done with sparsity")

        if isnothing(ximin)
            ximin = x₀
        end

        new{OB,ID,TD,PD}(process, objective, Δtb, Δtc, Nkb, Nkc, Nc,
            con_jacobian_sparsity,
            idx, sparsity_cache,
            x₀, ximin, x̄,
            final_constraint, hessian,
            cp,
            Δtb_min, Δtb_max,
            Δtc_min, Δtc_max)
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

    Nconb = Nstates + Neq + Nineq
    Nconc = Nstates
    Nconstr = Nconb * (Nkb * Nc) + Nconc * (Nkc * Nc - 1) + (final_constraint ? 2Nstates : Nstates)

    return (Nz=Nz, Nstates=Nstates, u=u, x=x, Δtb=Δtb, Δtc=Δtc, Nconstr=Nconstr, Nkb=Nkb, Nkc=Nkc, Nc=Nc, Nu=Nu, Neq=Neq, Nineq=Nineq, Nconb=Nconb, Nconc=Nconc)
end

function combined_dynamics!(f, x, u, process::Process{ID,TD,PD}, t, zi) where {ID,TD,PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics

    s = view(x, 1:Ns(td))
    α = view(x, (Ns(td)+1):(Ns(td)+Nα(pd)))
    r = view(x, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    ds = view(f, 1:Ns(td))
    dα = view(f, (Ns(td)+1):(Ns(td)+Nα(pd)))
    dr = view(f, (Ns(td)+Nα(pd)+1):(Ns(td)+Nα(pd)+Nr(id)))

    dynamics_function!(td, ds, s, t)
    input_function!(id, ds, r, u, t, zi) # always call second, additive
    dynamics_function!(pd, td, dα, α, s, t)
    dynamics_function!(id, dr, s, r, u, t, zi)
end

function collocation_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics
    Nx = length(xₖ)
    nu = length(uₖ)

    fₖ_cache = cp.fₖ_cache
    fₖ₊₁_cache = cp.fₖ₊₁_cache
    fₘ_cache = cp.fₘ_cache
    xₘ_cache = cp.xₘ_cache
    ẋₘ_cache = cp.ẋₘ_cache

    fₖ_cache_lock = cp.fₖ_cache_lock
    fₖ₊₁_cache_lock = cp.fₖ₊₁_cache_lock
    fₘ_cache_lock = cp.fₘ_cache_lock
    xₘ_cache_lock = cp.xₘ_cache_lock
    ẋₘ_cache_lock = cp.ẋₘ_cache_lock

    thread::Int = Threads.threadid()
    # lock(fₖ_cache_lock)
    fₖ = get!(fₖ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}
    # unlock(fₖ_cache_lock)
    # lock(fₖ₊₁_cache_lock)
    fₖ₊₁ = get!(fₖ₊₁_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}
    # unlock(fₖ₊₁_cache_lock)
    # lock(fₘ_cache_lock)
    fₘ = get!(fₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}
    # unlock(fₘ_cache_lock)
    # lock(xₘ_cache_lock)
    xₘ = get!(xₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}
    # unlock(xₘ_cache_lock)
    # lock(ẋₘ_cache_lock)
    ẋₘ = get!(ẋₘ_cache, (T, thread)) do
        zeros(T, Nx)
    end::Vector{T}
    # unlock(ẋₘ_cache_lock)

    combined_dynamics!(fₖ, xₖ, uₖ, process, t, zi)
    combined_dynamics!(fₖ₊₁, xₖ₊₁, uₖ, process, t + Δt[1], zi)

    xₘ .= @. 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    ẋₘ .= @. (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    combined_dynamics!(fₘ, xₘ, uₖ, process, t + Δt[1] / 2.0, zi)
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

function total_build_constraint!(process::Process{ID,TD,PD}, idx, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    Nx, Neq, Nineq = idx.Nstates, idx.Neq, idx.Nineq
    r_colloc = view(r, 1:Nx)
    r_eq = view(r, (Nx+1):(Nx+Neq))
    r_ineq = view(r, (Nx+Neq+1):(Nx+Neq+Nineq))

    collocation_constraint!(process, r_colloc, xₖ, uₖ, xₖ₊₁, Δt, t, zi, cp)
    equality_constraint!(process, r_eq, xₖ, uₖ, t, zi)
    inequality_constraint!(process, r_ineq, xₖ, uₖ, t, zi)
end

function total_cooling_constraint!(process::Process{ID,TD,PD}, idx, r::AbstractVector{T}, xₖ, xₖ₊₁, Δt, t, zi, cp::CachePackage) where {T,ID,TD,PD}
    ui = input_idle(process.input_dynamics)

    collocation_constraint!(process, r, xₖ, ui, xₖ₊₁, Δt, t, zi, cp)
end

function constraints!(process::Process, c, z, idx, xᵢ, cp::CachePackage; xf=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false)
    # Collocation, Initial state, Final State
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Nx, nu = idx.Nstates, idx.Nu
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)
    Nconb, Nconc = idx.Nconb, idx.Nconc

    # Initial state constraint
    @. c[1:Nx] = z[idx.x[1][1]] - xᵢ

    ic(c, k) = Nx + (c - 1) * (Nkb * Nconb + Nkc * Nconc) + (k > Nkb ? Nkb * Nconb + (k - Nkb - 1) * Nconc : (k - 1) * Nconb)
    t = 0.0
    for cyc in 1:Nc
        if cyc > 1
            if free_cool_time
                Δtc = z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end
            zi = kc2zi(Nkb + Nkc, cyc - 1, idx)

            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]
            i = ic(cyc - 1, Nkb + Nkc)
            r = @view c[(i+1):(i+Nconc)]

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t, zi, cp)
            # t += Δtc # TEMPORARY
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = z[idx.Δtb[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            i = ic(cyc, k)
            r = @view c[(i+1):(i+Nconb)]

            total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t, zi, cp)
            # t += Δtb # TEMPORARY
        end

        k = Nkb
        if free_build_time
            Δtb = z[idx.Δtb[cyc][k]]
        end
        zi = kc2zi(k, cyc, idx)

        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)
        i = ic(cyc, k)
        r = @view c[(i+1):(i+Nconb)]

        total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb, t, zi, cp)
        # t += Δtb # TEMPORARY

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = z[idx.Δtc[cyc][k]]
            end
            zi = kc2zi(k, cyc, idx)

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            i = ic(cyc, k)
            r = @view c[(i+1):(i+Nconc)]

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t, zi, cp)
            # t += Δtc # TEMPORARY
        end

    end

    # Final state constraint
    if final_constraint
        @. c[(end-Nx+1):(end)] = z[idx.x[Nc][Nkb+Nkc]] - xf
    end


end

function constraint_jacobian!(process::Process, jac, z, idx, sparsity_cache, cp::CachePackage; Δtb=nothing, Δtc=nothing, final_constraint=false, prob=nothing)
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Nx = idx.Nstates
    nu = idx.Nu
    Nconb, Nconc = idx.Nconb, idx.Nconc

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

    # Initial state constraint jacobian
    jac[1:Nx] .= 1

    Nconbjac = lJb∂xₖ + lJb∂uₖ + lJb∂xₖ₊₁ + lJb∂uₖ₊₁ + lJb∂Δt
    Nconcjac = lJc∂xₖ + lJc∂xₖ₊₁ + lJc∂Δt

    ic(c, k) = 1 + Nx + (c - 1) * (Nkb * Nconbjac + Nkc * Nconcjac) + (k > Nkb ? Nkb * Nconbjac + (k - Nkb - 1) * Nconcjac : (k - 1) * Nconbjac)
    t = 0.0

    for cyc in 1:Nc

        if cyc > 1
            i = ic(cyc - 1, Nkb + Nkc)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end
            zi = kc2zi(Nkb + Nkc, cyc - 1, idx)

            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            # uₖ = input_idle(process.input_dynamics)
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

        for k in 1:(Nkb-1)
            i = ic(cyc, k)
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

        k = Nkb
        i = ic(cyc, k)
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

        for k in (Nkb+1):(Nkb+Nkc-1)
            i = ic(cyc, k)
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

    # Final state constraint jacobian
    if final_constraint
        jac[(end-Nx+1):(end)] .= 1
    end

    # res = zeros(idx.Nconstr, idx.Nz)
    # rp = zeros(idx.Nconstr)
    # ForwardDiff.jacobian!(res, (r, z) -> constraints!(process, r, z, idx, prob.x₀, cp; xf=prob.x̄, Δtb=Δtb, Δtc=Δtc), rp, z)
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

function constraint_jacobian_sparsity(idx, process::Process, cp::CachePackage; Δtb=nothing, Δtc=nothing, final_constraint=false)
    Nx, Nu = idx.Nstates, idx.Nu
    Nkb, Nkc, Nc = idx.Nkb, idx.Nkc, idx.Nc
    Nconstr = idx.Nconstr
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

    NΔtb = free_build_time ? 1 : 0
    NΔtc = free_cool_time ? 1 : 0

    total_structure = Vector{Tuple{Int,Int}}()

    # Initial state constraint
    append!(total_structure, collect(zip(collect(1:Nx), collect(idx.x[1][1]))))
    row_offset = Nx

    println("Entering loop")
    for cyc in 1:Nc
        if cyc > 1
            r, c, _ = findnz(conc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc-1][Nkb+Nkc][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(conc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtc[cyc-1][Nkb+Nkc][1] .- 1)))
            end

            r, c, _ = findnz(conc_∂xₖ₊₁_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc][1][1] .- 1)))

            row_offset += Nconc
        end

        for k in 1:(Nkb-1)
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

        k = Nkb
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

        for k in (Nkb+1):(Nkb+Nkc-1)
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

    # Final state constraint
    if final_constraint
        append!(total_structure, collect(zip(collect((Nconstr-Nx+1):(Nconstr)), collect(idx.x[Nc][Nkb+Nkc]))))
    end

    # @show total_structure
    # display(sparse(rows, cols, trues(length(cols))))
    # total_structure = collect(zip(rows, cols))
    sparsity_cache = conb_∂xₖ_sparsity, conb_∂uₖ_sparsity, conb_∂xₖ₊₁_sparsity, conb_∂uₖ₊₁_sparsity, conb_∂Δt_sparsity, conc_∂xₖ_sparsity, conc_∂xₖ₊₁_sparsity, conc_∂Δt_sparsity, conb_∂uₖ_color, conb_∂xₖ_color, conb_∂xₖ₊₁_color, conb_∂uₖ₊₁_color, conb_∂Δt_color, conc_∂xₖ_color, conc_∂xₖ₊₁_color, conc_∂Δt_color, jac_cache_conb_∂Δt, jac_cache_conb_∂xₖ, jac_cache_conb_∂uₖ, jac_cache_conb_∂xₖ₊₁, jac_cache_conb_∂uₖ₊₁, jac_cache_conc_∂Δt, jac_cache_conc_∂xₖ, jac_cache_conc_∂xₖ₊₁
    return total_structure, sparsity_cache
end

function MOI.eval_objective(prob::AdditiveProblem, z)
    return cost(prob.objective, z, prob.idx)
end

function MOI.eval_objective_gradient(prob::AdditiveProblem, grad_f, z)
    gradient(prob.objective, grad_f, z, prob.idx)
end

function MOI.eval_constraint(prob::AdditiveProblem, c, z)
    constraints!(prob.process, c, z, prob.idx, prob.x₀, prob.cp, xf=prob.x̄, Δtb=prob.Δtb, Δtc=prob.Δtc, final_constraint=prob.final_constraint)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.sparsity_cache, prob.cp, Δtb=prob.Δtb, Δtc=prob.Δtc, final_constraint=prob.final_constraint, prob=prob)
end

function MOI.hessian_lagrangian_structure(prob::AdditiveProblem)
    return objective_hessian_structure(prob.objective, prob.idx)
end

function MOI.eval_hessian_lagrangian(prob::AdditiveProblem, H, z, σ, μ)
    objective_hessian_values(prob.objective, prob.idx, H, z)
    H .*= σ
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
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing, λ₀=nothing, ug=nothing, xg=nothing, solv="ma97")

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["hsllib"] = HSL_jll.libhsl_path
    solver.options["linear_solver"] = solv

    if solv == "ma77"
        solver.options["ma77_print_level"] = 2
    end

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
                    z₀[idx.Δtb[c][k]] = (problem.Δtb_min + problem.Δtb_max) / 2
                end
                z₀[idx.x[c][k]] .= isnothing(xg) ? problem.x̄ : xg
                z₀[idx.u[c][k]] .= isnothing(ug) ? input_min(id) : ug
            end

            for k in (Nkb+1):(Nkb+Nkc)
                if isnothing(problem.Δtc)
                    z₀[idx.Δtc[c][k]] = (problem.Δtc_min + problem.Δtc_max) / 2
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
    gr = zeros(Nz)
    jt = zeros(length(problem.constraint_jacobian_sparsity))
    μ0 = ones(Nconstr)

    @show Nz
    println("Checking objective function...")
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @time MOI.eval_objective(problem, z₀)
    @show MOI.eval_objective(problem, z₀)
    println("Checking objective gradient...")
    @time MOI.eval_objective_gradient(problem, gt, z₀)
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
        @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
    end

    println("Checking constraint function...")
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @time MOI.eval_constraint(problem, ct, z₀)
    @show maximum(abs.(ct))
    println("Checking constraint jacobian...")
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)
    @time MOI.eval_constraint_jacobian(problem, jt, z₀)

    ncf = problem.final_constraint ? Nx : 0
    c_lb = repeat([zeros(Nx + Neq); ineq_min(id)], Nkb)
    c_ub = repeat([zeros(Nx + Neq); ineq_max(id)], Nkb)
    c_lc = repeat(zeros(Nx), Nkc - 1)
    c_uc = repeat(zeros(Nx), Nkc - 1)

    c_l_cyc = vcat(c_lb, c_lc)
    c_u_cyc = vcat(c_ub, c_uc)

    c_l = vcat(problem.ximin .- problem.x₀, c_l_cyc, repeat(vcat(zeros(Nx), c_l_cyc), Nc - 1), zeros(ncf))
    c_u = vcat(zeros(Nx), c_u_cyc, repeat(vcat(zeros(Nx), c_u_cyc), Nc - 1), zeros(ncf))

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
                MOI.add_constraint(solver, Δtb, MOI.LessThan(problem.Δtb_max))
                MOI.add_constraint(solver, Δtb, MOI.GreaterThan(problem.Δtb_min))
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
                MOI.add_constraint(solver, Δtc, MOI.LessThan(problem.Δtc_max))
                MOI.add_constraint(solver, Δtc, MOI.GreaterThan(problem.Δtc_min))
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

    Δt = vcat([[isnothing(problem.Δtb) ? [result[idx.Δtb[c][k]] for k in 1:Nkb] : problem.Δtb * ones(Nkb)
        isnothing(problem.Δtc) ? [result[idx.Δtc[c][k]] for k in (Nkb+1):(Nkb+Nkc)] : problem.Δtc * ones(Nkc)
    ] for c in 1:Nc]...)

    return result, X, U, Δt, λ
end

end