module ADDOPT

@time using MathOptInterface, Ipopt
@time using LinearAlgebra, ForwardDiff
@time using Symbolics, SparseArrays, SparseDiffTools
@time using SparseArrays: findnz
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
include("initial_guess.jl")

fₖ_cache = Dict{DataType,Any}()
fₖ₊₁_cache = Dict{DataType,Any}()
fₘ_cache = Dict{DataType,Any}()
xₘ_cache = Dict{DataType,Any}()
uₘ_cache = Dict{DataType,Any}()
ẋₘ_cache = Dict{DataType,Any}()

struct AdditiveProblem{OB<:Objective, ID<:InputDynamics, TD<:TransferDynamics, PD<:PropertyDynamics} <: MOI.AbstractNLPEvaluator
    process::Process{ID, TD, PD}

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

    function AdditiveProblem(process::Process{ID, TD, PD}, objective::OB,  
        Nkb, Nkc, Nc, x₀; x̄=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false) where {OB<:Objective, ID<:InputDynamics, TD<:TransferDynamics, PD<:PropertyDynamics}
        id, td, pd = process.input_dynamics, process.transfer_dynamics, process.property_dynamics
        idx = generate_z_indices(Nkb, Nkc, Nc, Nu(id), Nr(id), Ns(td), Nα(pd), Δtb, Δtc, final_constraint)

        con_jacobian_sparsity, sparsity_cache = constraint_jacobian_sparsity(idx, process; Δtb=Δtb, Δtc=Δtc, final_constraint=final_constraint)

        new{OB, ID, TD, PD}(process, objective, Δtb, Δtc, Nkb, Nkc, Nc,
            con_jacobian_sparsity,
            idx, sparsity_cache, 
            x₀, x̄,
            final_constraint)
    end
end

function generate_z_indices(Nkb, Nkc, Nc, Nu, Nr, Ns, Nα, Δtb, Δtc, final_constraint)
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)
   
    Nstates = Ns + Nα + Nr 
    StatesPerCycle = Nstates * (Nkb+Nkc)
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
    
    x = [[(1:Nstates) .+ (Nstates*(k-1) +StatesPerCycle*(c-1)) for k in 1:(Nkb+Nkc)] for c in 1:Nc]
    u = [[(1:Nu) .+ (Nu*(k-1) +InputsPerCycle*(c-1) + Nstates_total) for k in 1:Nkb] for c in 1:Nc]
    Δtb = free_build_time ? [[k + (Nstates_total + Nu_total + ΔtbPerCycle*(c-1)) for k in 1:Nkb] for c in 1:Nc] : nothing # z[Δtb[c][k]] gives Δtb_(c,k), scalar
    Δtc = free_cool_time ? [vcat([0 for k in 1:Nkb],[k + (Nstates_total + Nu_total + NΔtb_total + ΔtcPerCycle*(c-1)) for k in 1:Nkc]) for c in 1:Nc] : nothing

    Nconstr = Nstates * ((Nkb + Nkc) * Nc - 1) + (final_constraint ? 2Nstates : Nstates)

    return (Nz=Nz, Nstates=Nstates, u=u, x=x, Δtb=Δtb, Δtc=Δtc, Nconstr=Nconstr, Nkb=Nkb, Nkc=Nkc, Nc=Nc, Nu=Nu)
end

function combined_dynamics!(f, x, u, process::Process{ID, TD, PD}) where {ID, TD, PD}
    td, pd, id = process.transfer_dynamics, process.property_dynamics, process.input_dynamics    
    # Nknotvals = Nstates  + Nu + NΔtb
    # Npercycle = (Nknotvals * Nkb + 1)
    # Nz = Npercycle * Nc #+ Nstates # Last term is for state after final coolingprocess.property_dynamics, process.input_dynamics

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

function collocation_constraint!(process::Process{ID, TD, PD}, r::AbstractVector{T}, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt; fₖ_cache::Dict{DataType,Any}=fₖ_cache, fₖ₊₁_cache::Dict{DataType,Any}=fₖ₊₁_cache, fₘ_cache::Dict{DataType,Any}=fₘ_cache, xₘ_cache::Dict{DataType,Any}=xₘ_cache, uₘ_cache::Dict{DataType,Any}=uₘ_cache, ẋₘ_cache::Dict{DataType,Any}=ẋₘ_cache) where {T, ID, TD, PD}
    Nx = length(xₖ)
    nu = length(uₖ)
    uₖ₊₁ = uₖ

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

    combined_dynamics!(fₖ, xₖ, uₖ, process)
    combined_dynamics!(fₖ₊₁, xₖ₊₁, uₖ₊₁, process)

    xₘ .= @. 0.5 * (xₖ + xₖ₊₁) + (Δt[1] / 8.0) * (fₖ - fₖ₊₁)
    uₘ .= @. 0.5 * (uₖ + uₖ₊₁)
    ẋₘ .= @. (3 / (2 * Δt[1])) * (xₖ₊₁ - xₖ) - 0.25 * (fₖ + fₖ₊₁)

    combined_dynamics!(fₘ, xₘ, uₘ, process)

    r[:] .= fₘ .- ẋₘ
end

function constraints!(process::Process, c, z, idx, xᵢ; xf=nothing, Δtb=nothing, Δtc=nothing, final_constraint=false)
    # Collocation, Initial state, Final State
    Nc, Nkb, Nkc = idx.Nc, idx.Nkb, idx.Nkc
    Nx, nu = idx.Nstates, idx.Nu
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    i = 0
    for cyc in 1:Nc
        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end

            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]
            uₖ₊₁ = @view z[idx.u[cyc][1]]
            r = @view c[(i+1):(i+Nx)]
            i += Nx

            collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtc)
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]
            r = @view c[(i+1):(i+Nx)]
            i += Nx

            collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
        end

        k = Nkb
        if free_build_time
            Δtb = @view z[idx.Δtb[cyc][k]]
        end

        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)
        r = @view c[(i+1):(i+Nx)]
        i += Nx

        collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb)

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end

            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            r = @view c[(i+1):(i+Nx)]
            i += Nx

            collocation_constraint!(process, r, xₖ, uᵢ, xₖ₊₁, uᵢ, Δtc)
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
    colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color = sparsity_cache
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    colloc_∂Δt(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, uₖ₊₁, X), Δt, colorvec=colloc_∂Δt_color)
    colloc_∂xₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, X, uₖ, xₖ₊₁, uₖ₊₁, Δt), xₖ, colorvec=colloc_∂xₖ_color)
    colloc_∂uₖ(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, X, xₖ₊₁, uₖ₊₁, Δt), uₖ, colorvec=colloc_∂uₖ_color)
    colloc_∂xₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, X, uₖ₊₁, Δt), xₖ₊₁, colorvec=colloc_∂xₖ₊₁_color)
    colloc_∂uₖ₊₁(J, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δt) = forwarddiff_color_jacobian!(J, (r, X) -> collocation_constraint!(process, r, xₖ, uₖ, xₖ₊₁, X, Δt), uₖ₊₁, colorvec=colloc_∂uₖ₊₁_color)

    i = 1
    J∂xₖ = Float64.(sparse(colloc_∂xₖ_sparsity))
    J∂uₖ = Float64.(sparse(colloc_∂uₖ_sparsity))
    J∂xₖ₊₁ = Float64.(sparse(colloc_∂xₖ₊₁_sparsity))
    J∂uₖ₊₁ = Float64.(sparse(colloc_∂uₖ₊₁_sparsity))
    J∂Δt = Float64.(sparse(colloc_∂Δt_sparsity))

    for cyc in 1:Nc

        if cyc > 1
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc-1][Nkb+Nkc]]
            end
            xₖ = @view z[idx.x[cyc-1][Nkb+Nkc]]
            uₖ = input_idle(process.input_dynamics)
            xₖ₊₁ = @view z[idx.x[cyc][1]]
            uₖ₊₁ = @view z[idx.u[cyc][1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtc)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtc)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtc)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ₊₁(J∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtc)
            _, _, vals = findnz(J∂uₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end

        for k in 1:(Nkb-1)
            if free_build_time
                Δtb = @view z[idx.Δtb[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            uₖ = @view z[idx.u[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]
            uₖ₊₁ = @view z[idx.u[cyc][k+1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
            _, _, vals = findnz(J∂uₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_build_time
                colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            colloc_∂uₖ₊₁(J∂uₖ₊₁, xₖ, uₖ, xₖ₊₁, uₖ₊₁, Δtb)
            _, _, vals = findnz(J∂uₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end

        k = Nkb
        if free_build_time
            Δtb = @view z[idx.Δtb[cyc][k]]
        end
        xₖ = @view z[idx.x[cyc][k]]
        uₖ = @view z[idx.u[cyc][k]]
        xₖ₊₁ = @view z[idx.x[cyc][k+1]]
        uᵢ = input_idle(process.input_dynamics)

        colloc_∂xₖ(J∂xₖ, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb)
        _, _, vals = findnz(J∂xₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        colloc_∂uₖ(J∂uₖ, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb)
        _, _, vals = findnz(J∂uₖ)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        if free_build_time
            colloc_∂Δt(J∂Δt, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb)
            _, _, vals = findnz(J∂Δt)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end

        colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uₖ, xₖ₊₁, uᵢ, Δtb)
        _, _, vals = findnz(J∂xₖ₊₁)
        view(jac, i:(i+length(vals)-1)) .= vals
        i += length(vals)

        for k in (Nkb+1):(Nkb+Nkc-1)
            if free_cool_time
                Δtc = @view z[idx.Δtc[cyc][k]]
            end
            xₖ = @view z[idx.x[cyc][k]]
            xₖ₊₁ = @view z[idx.x[cyc][k+1]]

            colloc_∂xₖ(J∂xₖ, xₖ, uᵢ, xₖ₊₁, uᵢ, Δtc)
            _, _, vals = findnz(J∂xₖ)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)

            if free_cool_time
                colloc_∂Δt(J∂Δt, xₖ, uᵢ, xₖ₊₁, uᵢ, Δtc)
                _, _, vals = findnz(J∂Δt)
                view(jac, i:(i+length(vals)-1)) .= vals
                i += length(vals)
            end

            colloc_∂xₖ₊₁(J∂xₖ₊₁, xₖ, uᵢ, xₖ₊₁, uᵢ, Δtc)
            _, _, vals = findnz(J∂xₖ₊₁)
            view(jac, i:(i+length(vals)-1)) .= vals
            i += length(vals)
        end
    end

    if final_constraint
        jac[(end-2Nx+1):(end-Nx)] .= 1
    end
    jac[(end-Nx+1):(end)] .= 1

    # res = zeros(idx.Nconstr, idx.Nz)
    # rp = zeros(idx.Nconstr)
    # ForwardDiff.jacobian!(res, (r,z) -> constraints!(process, r, z, idx, [293.15; 0.0]; xf=[298.15; 0.8], Δtb=Δtb, Δtc=Δtc), rp, z)
    # display("reference")
    # display(res)

    # rs = [r for (r,c) in prob.constraint_jacobian_sparsity]
    # cs = [c for (r,c) in prob.constraint_jacobian_sparsity]
    # display("actual")
    # display(Matrix(sparse(rs, cs, jac)))
end

function constraint_jacobian_sparsity(idx, process::Process; Δtb=nothing, Δtc=nothing, final_constraint=false)
    Nx, Nu = idx.Nstates, idx.Nu
    Nkb, Nkc, Nc = idx.Nkb, idx.Nkc, idx.Nc
    Nconstr = idx.Nconstr
    free_build_time = isnothing(Δtb)
    free_cool_time = isnothing(Δtc)

    rd = ones(Symbolics.Num, Nx)
    xd1 = 2 * ones(Symbolics.Num, Nx)
    ud1 = 3 * ones(Symbolics.Num, Nu)
    xd2 = 4 * ones(Symbolics.Num, Nx)
    ud2 = 5 * ones(Symbolics.Num, Nu)
    Δtd = 6 * ones(Symbolics.Num, 1)

    colloc_∂xₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, X, ud1, xd2, ud2, Δtd), rd, xd1)
    display(colloc_∂xₖ_sparsity)
    colloc_∂uₖ_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, X, xd2, ud2, Δtd), rd, ud1)
    display(colloc_∂uₖ_sparsity)
    colloc_∂xₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, X, ud2, Δtd), rd, xd2)
    colloc_∂uₖ₊₁_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, xd2, X, Δtd), rd, ud2)
    colloc_∂Δt_sparsity = Symbolics.jacobian_sparsity((r, X) -> collocation_constraint!(process, r, xd1, ud1, xd2, ud2, X), rd, Δtd)
    if free_build_time || free_cool_time
        display(colloc_∂Δt_sparsity)
    end

    colloc_∂uₖ_color = matrix_colors(Float64.(colloc_∂uₖ_sparsity))
    colloc_∂xₖ_color = matrix_colors(Float64.(colloc_∂xₖ_sparsity))
    colloc_∂xₖ₊₁_color = matrix_colors(Float64.(colloc_∂xₖ₊₁_sparsity))
    colloc_∂uₖ₊₁_color = matrix_colors(Float64.(colloc_∂uₖ₊₁_sparsity))
    colloc_∂Δt_color = matrix_colors(Float64.(colloc_∂Δt_sparsity))

    NΔtb = free_build_time ? 1 : 0
    NΔtc = free_cool_time ? 1 : 0
    row_offset = 0
    rows = []
    cols = []
    for cyc in 1:Nc
        if cyc > 1
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc-1][Nkb+Nkc][1] .- 1)

            if free_cool_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(rows, r .+ row_offset)
                append!(cols, c .+ idx.Δtb[cyc-1][Nkb+Nkc][1] .- 1)
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc][1][1] .- 1)

            r, c, _ = findnz(colloc_∂uₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.u[cyc][1][1] .- 1)

            row_offset += Nx
        end

        for k in 1:(Nkb-1)
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc][k][1] .- 1)

            r, c, _ = findnz(colloc_∂uₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.u[cyc][k][1] .- 1)

            if free_build_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(rows, r .+ row_offset)
                append!(cols, c .+ idx.Δtb[cyc][k][1]  .- 1)
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc][k+1][1] .- 1)

            r, c, _ = findnz(colloc_∂uₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.u[cyc][k+1][1] .- 1)

            row_offset += Nx
        end

        k = Nkb
        r, c, _ = findnz(colloc_∂xₖ_sparsity)
        append!(rows, r .+ row_offset)
        append!(cols, c .+ idx.x[cyc][k][1] .- 1)

        r, c, _ = findnz(colloc_∂uₖ_sparsity)
        append!(rows, r .+ row_offset)
        append!(cols, c .+ idx.u[cyc][k][1] .- 1)

        if free_build_time
            r, c, _ = findnz(colloc_∂Δt_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.Δtb[cyc][k][1]  .- 1)
        end

        r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
        append!(rows, r .+ row_offset)
        append!(cols, c .+ idx.x[cyc][k+1][1] .- 1)

        row_offset += Nx

        for k in (Nkb+1):(Nkb+Nkc-1)
            r, c, _ = findnz(colloc_∂xₖ_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc][k][1] .- 1)

            if free_cool_time
                r, c, _ = findnz(colloc_∂Δt_sparsity)
                append!(rows, r .+ row_offset)
                append!(cols, c .+ idx.Δtc[cyc][k][1] .- 1)
            end

            r, c, _ = findnz(colloc_∂xₖ₊₁_sparsity)
            append!(rows, r .+ row_offset)
            append!(cols, c .+ idx.x[cyc][k+1][1] .- 1)

            row_offset += Nx
        end

    end

    if final_constraint
        append!(rows, collect((Nconstr-2Nx+1):(Nconstr-Nx)))
        append!(cols, collect(idx.x[Nc][Nkb+Nkc]))
    end

    append!(rows, collect((Nconstr-Nx+1):(Nconstr)))
    append!(cols, collect(idx.x[1][1]))

    display(sparse(rows, cols, trues(length(cols))))
    total_structure = collect(zip(rows, cols))
    sparsity_cache = colloc_∂xₖ_sparsity, colloc_∂uₖ_sparsity, colloc_∂xₖ₊₁_sparsity, colloc_∂uₖ₊₁_sparsity, colloc_∂Δt_sparsity, colloc_∂uₖ_color, colloc_∂xₖ_color, colloc_∂xₖ₊₁_color, colloc_∂uₖ₊₁_color, colloc_∂Δt_color
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
    constraint_jacobian!(prob.process, jac, z, prob.idx, prob.sparsity_cache, Δtb=prob.Δtb, Δtc=prob.Δtc, prob=prob, final_constraint=prob.final_constraint)
end

function MOI.hessian_lagrangian_structure(prob::AdditiveProblem)
    return hessian_structure(prob.objective, prob.idx)
end

function MOI.eval_hessian_lagrangian(prob::AdditiveProblem, H, z, σ, μ)
    hessian_values(prob.objective, prob.idx, H)
end

MOI.features_available(prob::AdditiveProblem) = [:Grad, :Jac, :Hess]
MOI.initialize(prob::AdditiveProblem, features) = nothing
MOI.jacobian_structure(prob::AdditiveProblem) = prob.constraint_jacobian_sparsity

function optimize_trajectory(problem::AdditiveProblem;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=500, z₀=nothing)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    idx = problem.idx
    Nz, Nconstr = idx.Nz, idx.Nconstr
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
                z₀[idx.x[c][k]] .= problem.x̄
                z₀[idx.u[c][k]] .= [5000.0]
                z₀[idx.x[c][k][1]] = 600
            end

            for k in (Nkb+1):(Nkb+Nkc)
                if isnothing(problem.Δtc)
                    z₀[idx.Δtc[c][k]] = 0.04
                end
                z₀[idx.x[c][k]] .= problem.x̄
                z₀[idx.x[c][k][1]] = 600
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

        for i in 1:lastindex(z)
            MOI.set(solver, MOI.VariablePrimalStart(), z[i], z₀[i])
        end
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    result = MOI.get(solver, MOI.VariablePrimal(), z)
    X = vcat([[result[idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
    U = vcat([vcat([result[idx.u[c][k]] for k in 1:Nkb],[input_idle(id) for k in 1:Nkc]) for c in 1:Nc]...)
    Δt = nothing
    if isnothing(problem.Δtb)
        Δt = [[result[idx.Δt[c][k]] for k in 1:Nkb] for c in 1:Nc]
    end

    return result, X, U, Δt
end

end