module ADDOPT

@time using PrettyTables
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
Nα(pd::PropertyDynamics) = 0
Ns(td::TransferDynamics) = 0

Nc_ineq(id::Dynamics) = 0
Nc_eq(id::Dynamics) = 0
Nc_ineq_inter(id::Dynamics) = 0

function equality_constraint!(id::InputDynamics, c::AbstractVector{Ty}, u, t) where {Ty}
end

function inequality_constraint!(id::InputDynamics, c::AbstractVector{Ty}, u, t) where {Ty}
end

ineq_min(id::Dynamics) = []
ineq_max(id::Dynamics) = []

function inequality_constraint_interstep!(id::InputDynamics, c::AbstractVector{Ty}, uₖ, t, uₖ₊₁) where {Ty}
end

ineq_inter_min(id::Dynamics) = []
ineq_inter_max(id::Dynamics) = []

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
        idx = generate_z_indices(Nkb, Nkc, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), Nc_eq(id), Nc_ineq(id), Nc_ineq_inter(id), Δtb, Δtc, final_constraint)

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

function generate_z_indices(Nkb, Nkc, Nc, Nu, Nr, Ns, Nα, Neq, Nineq, Nineq_inter, Δtb, Δtc, final_constraint)
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

    Nconb = Nstates + Neq + Nineq + Nineq_inter
    Nconc = Nstates
    Nconstr = Nconb * (Nkb * Nc) + Nconc * (Nkc * Nc - 1) + (final_constraint ? 2Nstates : Nstates)

    return (Nz=Nz, Nstates=Nstates, u=u, x=x, Δtb=Δtb, Δtc=Δtc, Nconstr=Nconstr, Nkb=Nkb, Nkc=Nkc, Nc=Nc, Nu=Nu, Neq=Neq, Nineq=Nineq, Nineq_inter=Nineq_inter, Nconb=Nconb, Nconc=Nconc)
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

function collocation_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, Δt, t; fₖ_cache::Dict{DataType,Any}=fₖ_cache, fₖ₊₁_cache::Dict{DataType,Any}=fₖ₊₁_cache, fₘ_cache::Dict{DataType,Any}=fₘ_cache, xₘ_cache::Dict{DataType,Any}=xₘ_cache, ẋₘ_cache::Dict{DataType,Any}=ẋₘ_cache) where {T,ID,TD,PD}
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
    r .= fₘ .- ẋₘ
end

function equality_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, t) where {T,ID,TD,PD}
    ns = Ns(process.transfer_dynamics)
    nα = Nα(process.property_dynamics)
    nr = Nr(process.input_dynamics)
    rₖ = view(xₖ, (ns+nα+1):(ns+nα+nr))

    equality_constraint!(process.input_dynamics, r, uₖ, t)
end

function inequality_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, t) where {T,ID,TD,PD}
    ns = Ns(process.transfer_dynamics)
    nα = Nα(process.property_dynamics)
    nr = Nr(process.input_dynamics)
    rₖ = view(xₖ, (ns+nα+1):(ns+nα+nr))

    inequality_constraint!(process.input_dynamics, r, uₖ, t)
end

function inequality_interstep_constraint!(process::Process{ID,TD,PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) where {T,ID,TD,PD}
    ns = Ns(process.transfer_dynamics)
    nα = Nα(process.property_dynamics)
    nr = Nr(process.input_dynamics)
    rₖ = view(xₖ, (ns+nα+1):(ns+nα+nr))
    rₖ₊₁ = view(xₖ₊₁, (ns+nα+1):(ns+nα+nr))

    inequality_constraint_interstep!(process.input_dynamics, r, uₖ, t, uₖ₊₁)
end

function total_build_constraint!(process::Process{ID,TD,PD}, idx, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) where {T,ID,TD,PD}
    Nx, Neq, Nineq, Nineq_inter = idx.Nstates, idx.Neq, idx.Nineq, idx.Nineq_inter
    r_colloc = view(r, 1:Nx)
    r_eq = view(r, (Nx+1):(Nx+Neq))
    r_ineq = view(r, (Nx+Neq+1):(Nx+Neq+Nineq))
    r_ineq_inter = view(r, (Nx+Neq+Nineq+1):(Nx+Neq+Nineq+Nineq_inter))

    collocation_constraint!(process, r_colloc, xₖ, uₖ, xₖ₊₁, Δt, t)
    equality_constraint!(process, r_eq, xₖ, uₖ, t)
    inequality_constraint!(process, r_ineq, xₖ, uₖ, t)
    inequality_interstep_constraint!(process, r_ineq_inter, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t)
end

function total_cooling_constraint!(process::Process{ID,TD,PD}, idx, r::AbstractVector{T}, xₖ, xₖ₊₁, Δt, t) where {T,ID,TD,PD}
    ui = input_idle(process.input_dynamics)

    collocation_constraint!(process, r, xₖ, ui, xₖ₊₁, Δt, t)
end

function constraints!(process::Process, c, z, idx, xᵢ; xf=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false)
    # Collocation, Initial state, Final State
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Nx, nu = idx.Nstates, idx.Nu
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)
    Nconb, Nconc = idx.Nconb, idx.Nconc

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
            r = @view c[(i+1):(i+Nconc)]

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t)
            i += Nconc

            t += Δtc
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            r = @view c[(i+1):(i+Nconb)]
            i += Nconb

            total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
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
        r = @view c[(i+1):(i+Nconb)]
        i += Nconb

        total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb, t)
        t += Δtb

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            r = @view c[(i+1):(i+Nconc)]
            i += Nconc

            total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, Δtc, t)
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
    Nx = idx.Nstates
    nu = idx.Nu
    Nconb, Nconc = idx.Nconb, idx.Nconc

    conb_∂xₖ_sparsity, conb_∂uₖ_sparsity, conb_∂xₖ₊₁_sparsity, conb_∂uₖ₊₁_sparsity, conb_∂Δt_sparsity, conc_∂xₖ_sparsity, conc_∂xₖ₊₁_sparsity, conc_∂Δt_sparsity, conb_∂uₖ_color, conb_∂xₖ_color, conb_∂xₖ₊₁_color, conb_∂uₖ₊₁_color, conb_∂Δt_color, conc_∂xₖ_color, conc_∂xₖ₊₁_color, conc_∂Δt_color = sparsity_cache
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    conb_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X, t), Δt, colorvec=conb_∂Δt_color)
    conb_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt, t), xₖ, colorvec=conb_∂xₖ_color)
    conb_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt, t), uₖ, colorvec=conb_∂uₖ_color)
    conb_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, X, uₖ₊₁, Δt, t), xₖ₊₁, colorvec=conb_∂xₖ₊₁_color)
    conb_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, X, Δt, t), uₖ₊₁, colorvec=conb_∂uₖ₊₁_color)

    conc_∂Δt(J, xₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, X, t), Δt, colorvec=conc_∂Δt_color)
    conc_∂xₖ(J, xₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, X, xₖ₊₁, Δt, t), xₖ, colorvec=conc_∂xₖ_color)
    conc_∂xₖ₊₁(J, xₖ, xₖ₊₁, Δt, t) = forwarddiff_color_jacobian!(J, (r, X) -> total_cooling_constraint!(process, idx, r, xₖ, X, Δt, t), xₖ₊₁, colorvec=conc_∂xₖ₊₁_color)

    i = 1
    Jb∂xₖ = Float64.(sparse(conb_∂xₖ_sparsity))
    Jb∂uₖ = Float64.(sparse(conb_∂uₖ_sparsity))
    Jb∂xₖ₊₁ = Float64.(sparse(conb_∂xₖ₊₁_sparsity))
    Jb∂uₖ₊₁ = Float64.(sparse(conb_∂uₖ₊₁_sparsity))
    Jb∂Δt = Float64.(sparse(conb_∂Δt_sparsity))

    Jc∂xₖ = Float64.(sparse(conc_∂xₖ_sparsity))
    Jc∂xₖ₊₁ = Float64.(sparse(conc_∂xₖ₊₁_sparsity))
    Jc∂Δt = Float64.(sparse(conc_∂Δt_sparsity))

    t = 0.0

    for cyc in 1:Nc

        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end
            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            # uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]

            conc_∂xₖ(Jc∂xₖ, xₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(Jc∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                conc_∂Δt(Jc∂Δt, xₖ, xₖ₊₁, Δtc, t)
                _, _, vals = findnz(Jc∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            conc_∂xₖ₊₁(Jc∂xₖ₊₁, xₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(Jc∂xₖ₊₁)
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
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]

            conb_∂xₖ(Jb∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
            _, _, vals = findnz(Jb∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            conb_∂uₖ(Jb∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
            _, _, vals = findnz(Jb∂uₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_build_time
                conb_∂Δt(Jb∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
                _, _, vals = findnz(Jb∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            conb_∂xₖ₊₁(Jb∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
            _, _, vals = findnz(Jb∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            conb_∂uₖ₊₁(Jb∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
            _, _, vals = findnz(Jb∂uₖ₊₁)
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
        uₖ₊₁ = input_idle(process.input_dynamics)

        conb_∂xₖ(Jb∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
        _, _, vals = findnz(Jb∂xₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        conb_∂uₖ(Jb∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
        _, _, vals = findnz(Jb∂uₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        if free_build_time
            conb_∂Δt(Jb∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
            _, _, vals = findnz(Jb∂Δt)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end

        conb_∂xₖ₊₁(Jb∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb, t)
        _, _, vals = findnz(Jb∂xₖ₊₁)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)
        t += Δtb

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]

            conc_∂xₖ(Jc∂xₖ, xₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(Jc∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                conc_∂Δt(Jc∂Δt, xₖ, xₖ₊₁, Δtc, t)
                _, _, vals = findnz(Jc∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            conc_∂xₖ₊₁(Jc∂xₖ₊₁, xₖ, xₖ₊₁, Δtc, t)
            _, _, vals = findnz(Jc∂xₖ₊₁)
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


    conb_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δtd, t), rdb, xₖ)
    conb_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δtd, t), rdb, uₖ)
    conb_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, X, uₖ₊₁, Δtd, t), rdb, xₖ₊₁)
    conb_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, X, Δtd, t), rdb, uₖ₊₁)
    conb_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_build_constraint!(process, idx, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X, t), rdb, Δtd)

    conc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, X, xₖ₊₁, Δtd, t), rdc, xₖ)
    conc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, X, Δtd, t), rdc, xₖ₊₁)
    conc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> total_cooling_constraint!(process, idx, r, xₖ, xₖ₊₁, X, t), rdc, Δtd)

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

    NΔtb = free_build_time ? 1 : 0
    NΔtc = free_cool_time ? 1 : 0
    row_offset = 0

    total_structure = Vector{Tuple{Int,Int}}()
    println("Entering loop")
    for cyc in 1:Nc
        if cyc > 1
            r, c, _ = findnz(conc_∂xₖ_sparsity)
            append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.x[cyc-1][Nkb+Nkc][1] .- 1)))

            if free_cool_time
                r, c, _ = findnz(conc_∂Δt_sparsity)
                append!(total_structure, collect(zip(r .+ row_offset, c .+ idx.Δtb[cyc-1][Nkb+Nkc][1] .- 1)))
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

    if final_constraint
        append!(total_structure, collect(zip(collect((Nconstr-2Nx+1):(Nconstr-Nx)), collect(idx.x[Nc][Nkb+Nkc]))))
    end

    append!(total_structure, collect(zip(collect((Nconstr-Nx+1):(Nconstr)), collect(idx.x[1][1]))))

    # display(sparse(rows, cols, trues(length(cols))))
    # total_structure = collect(zip(rows, cols))
    sparsity_cache = conb_∂xₖ_sparsity, conb_∂uₖ_sparsity, conb_∂xₖ₊₁_sparsity, conb_∂uₖ₊₁_sparsity, conb_∂Δt_sparsity, conc_∂xₖ_sparsity, conc_∂xₖ₊₁_sparsity, conc_∂Δt_sparsity, conb_∂uₖ_color, conb_∂xₖ_color, conb_∂xₖ₊₁_color, conb_∂uₖ₊₁_color, conb_∂Δt_color, conc_∂xₖ_color, conc_∂xₖ₊₁_color, conc_∂Δt_color
    return total_structure, sparsity_cache
end


function collocation_constraint_hessian_structure(process::Process{ID,TD,PD}, c, k, idx) where {ID,TD,PD}
    return []
end

function equality_constraint_hessian_structure(process::Process{ID,TD,PD}, c, k, idx) where {ID,TD,PD}
    Neq = idx.Neq
    structure = Vector{Tuple{Int,Int}}()

    for i in 1:Neq
        str = equality_constraint_hessian_structure(process.input_dynamics, i)
        str = [(row + idx.u[c][k][1] - 1, col + idx.u[c][k][1] - 1) for (row, col) in str]
        append!(structure, str)
    end

    return structure
end

function inequality_constraint_hessian_structure(process::Process{ID,TD,PD}, c, k, idx) where {ID,TD,PD}
    Nineq = idx.Nineq
    structure = Vector{Tuple{Int,Int}}()

    for i in 1:Nineq
        str = inequality_constraint_hessian_structure(process.input_dynamics, i)
        str = [(row + idx.u[c][k][1] - 1, col + idx.u[c][k][1] - 1) for (row, col) in str]
        append!(structure, str)
    end

    return structure
end

function inequality_interstep_constraint_hessian_structure(process::Process{ID,TD,PD}, c, k, idx) where {ID,TD,PD}
    Nineq_inter = idx.Nineq_inter
    structure = Vector{Tuple{Int,Int}}()

    for i in 1:Nineq_inter
        str = inequality_interstep_constraint_hessian_structure(process.input_dynamics, i)
        str = [(row + idx.u[c][k][1] - 1, col + idx.u[c][k][1] - 1) for (row, col) in str]
        append!(structure, str)
    end

    return structure
end

function total_build_hessian_structure(process::Process{ID,TD,PD}, idx, c, k) where {ID,TD,PD}
    if k < idx.Nkb

        return vcat(
            collocation_constraint_hessian_structure(process, c, k, idx),
            equality_constraint_hessian_structure(process, c, k, idx),
            inequality_constraint_hessian_structure(process, c, k, idx),
            inequality_interstep_constraint_hessian_structure(process, c, k, idx))
    else
        return vcat(
            collocation_constraint_hessian_structure(process, c, k, idx),
            equality_constraint_hessian_structure(process, c, k, idx),
            inequality_constraint_hessian_structure(process, c, k, idx))
    end
end

function total_cool_hessian_structure(process::Process{ID,TD,PD}, idx, c, k) where {ID,TD,PD}
    return collocation_constraint_hessian_structure(process, c, k, idx)
end

function constraint_hessian_structure(process::Process, idx)
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    structure = Vector{Tuple{Int,Int}}()

    for c in 1:Nc
        if c > 1
            append!(structure, total_cool_hessian_structure(process, idx, c, k))
        end

        for k in 1:(Nkb-1)
            append!(structure, total_build_hessian_structure(process, idx, c, k))
        end

        k = Nkb
        append!(structure, total_build_hessian_structure(process, idx, c, k))

        for k in (Nkb+1):(Nkb+Nkc-1)
            append!(structure, total_cool_hessian_structure(process, idx, c, k))
        end
    end

    return structure
end


function collocation_constraint_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j) where {T,ID,TD,PD}
    j += idx.Nstates
    return i, j
end

function equality_constraint_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j) where {T,ID,TD,PD}
    Neq = idx.Neq

    for k in 1:Neq
        i, j = equality_constraint_hessian_values(process.input_dynamics, H, μ, i, j, k)
    end

    return i, j
end

function inequality_constraint_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j) where {T,ID,TD,PD}
    Nineq = idx.Nineq

    for k in 1:Nineq
        i, j = inequality_constraint_hessian_values(process.input_dynamics, H, μ, i, j, k)
    end

    return i, j
end

function inequality_interstep_constraint_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j) where {T,ID,TD,PD}
    Nineq_inter = idx.Nineq_inter

    for k in 1:Nineq_inter
        i, j = inequality_interstep_constraint_hessian_values(process.input_dynamics, H, μ, i, j, k)
    end

    return i, j
end

function total_build_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j, k) where {T,ID,TD,PD}
    i, j = collocation_constraint_hessian_values(process, idx, H, μ, i, j)
    i, j = equality_constraint_hessian_values(process, idx, H, μ, i, j)
    i, j = inequality_constraint_hessian_values(process, idx, H, μ, i, j)
    if k < idx.Nkb
        i, j = inequality_interstep_constraint_hessian_values(process, idx, H, μ, i, j)
    else
        j += idx.Nineq_inter
    end

    return i, j
end

function total_cool_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ, i, j) where {T,ID,TD,PD}
    return collocation_constraint_hessian_values(process, idx, H, μ, i, j)
end

function constraint_hessian_values(process::Process{ID,TD,PD}, idx, H::AbstractVector{T}, μ) where {T,ID,TD,PD}
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    i = 1
    j = 1

    for c in 1:Nc
        if c > 1
            i, j = total_cool_hessian_values(process, idx, H, μ, i, j)
        end

        for k in 1:(Nkb-1)
            i, j = total_build_hessian_values(process, idx, H, μ, i, j, k)
        end

        k = Nkb
        i, j = total_build_hessian_values(process, idx, H, μ, i, j, k)

        for k in (Nkb+1):(Nkb+Nkc-1)
            if k == Nkb + Nkc - 1 && c == Nc
                break
            end

            i, j = total_cool_hessian_values(process, idx, H, μ, i, j)
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
    constraints!(prob.process, c, z, prob.idx, prob.x₀, xf=prob.x̄, Δtb=prob.Δtb, Δtc=prob.Δtc, final_constraint=prob.final_constraint)
end

function MOI.eval_constraint_jacobian(prob::AdditiveProblem, jac, z)
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.sparsity_cache, Δtb=prob.Δtb, Δtc=prob.Δtc, prob=prob, final_constraint=prob.final_constraint)
end

function MOI.hessian_lagrangian_structure(prob::AdditiveProblem)
    obj_struct = objective_hessian_structure(prob.objective, prob.idx)
    con_struct = []#constraint_hessian_structure(prob.process, prob.idx)
    return vcat(obj_struct, con_struct)
end

function MOI.eval_hessian_lagrangian(prob::AdditiveProblem, H, z, σ, μ)
    # Nz = prob.idx.Nz
    H_obj = H
    # H_obj = @view H[1:Nz]
    # H_con = @view H[(Nz+1):end]

    objective_hessian_values(prob.objective, prob.idx, H_obj)
    H_obj .*= σ

    # constraint_hessian_values(prob.process, prob.idx, H_con, μ)
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
                z₀[idx.u[c][k]] .= isnothing(ug) ? input_min(id) : ug#(input_min(id) .+ input_max(id)) ./ 2
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
    μ0 = ones(Nconstr)
    structure = MOI.hessian_lagrangian_structure(problem)
    rs = [r for (r, c) in structure]
    cs = [c for (r, c) in structure]
    @show maximum(rs)
    @show minimum(rs)
    @show maximum(cs)
    @show minimum(cs)
    @show Nz
    H0 = zeros(length(structure))
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
    println("Checking lagrangian hessian...")
    @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
    @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)
    @time MOI.eval_hessian_lagrangian(problem, H0, z₀, 1.0, μ0)

    ncf = problem.final_constraint ? 2Nx : Nx
    c_lb = repeat([zeros(Nx + Neq); ineq_min(id); ineq_inter_min(id)], Nkb)
    c_ub = repeat([zeros(Nx + Neq); ineq_max(id); ineq_inter_max(id)], Nkb)
    c_lc = repeat(zeros(Nx), Nkc - 1)
    c_uc = repeat(zeros(Nx), Nkc - 1)

    c_l_cyc = vcat(c_lb, c_lc)
    c_u_cyc = vcat(c_ub, c_uc)

    c_l = vcat(c_l_cyc, repeat(vcat(zeros(Nx), c_l_cyc), Nc - 1), zeros(ncf))
    c_u = vcat(c_u_cyc, repeat(vcat(zeros(Nx), c_u_cyc), Nc - 1), zeros(ncf))

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