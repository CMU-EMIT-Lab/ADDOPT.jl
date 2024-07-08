
struct Problem{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T},D<:Dynamics{T},C<:Cost{T},H<:Constraint{T}}
    x₀::V                        # Initial condition
    process::Process{T,V,D}       # Underlying process
    z::Trajectory{T,V}
    z̄::Trajectory{T,V}
    v::ConstraintTrajectory{T,V}
    v̄::ConstraintTrajectory{T,V}

    costs::Vector{C}             # Stage cost for each timestep, Nk long
    constraints::Vector{H} # Constraints applied at each timestep, Nk long

    A::Vector{M}   # Cache of dynamics matrices (state Jacobians)
    B::Vector{M}   # Cache of input matrices (input Jacobians)

    cx::Vector{M}  # Cache of constraint state Jacobians
    cu::Vector{M}  # Cache of constraint input Jacobians

    lx::Vector{V}  # Cache of cost state gradients
    lu::Vector{V}  # Cache of cost input gradients
    lxx::Vector{M} # Cache of cost state Hessians
    luu::Vector{M} # Cache of cost input Hessians
    lux::Vector{M} # Cache of cost state-input Hessians

    Qx::Vector{V}  # Cache of cost-to-go state gradients
    Qu::Vector{V}  # Cache of cost-to-go input gradients
    Qxx::Vector{M} # Cache of cost-to-go state Hessians
    Quu::Vector{M} # Cache of cost-to-go input Hessians
    Qux::Vector{M} # Cache of cost-to-go state-input Hessians
    AtP::Vector{M} # Cache of A' * P 
    BtP::Vector{M} # Cache of A' * P
    KtQu::Vector{M} # Cache of K' * Quu
    λIμc::Vector{V}     # Cache of λ + Iμ*c
    cxIμ::Vector{M} # Cache of cx' * Iμ
    cuIμ::Vector{M} # Cache of cu' * Iμ
    Quus::Vector{M} # Cache of cost-to-go input Hessians
    Iρ::Vector{Diagonal{T,V}}

    P::Vector{M}   # Cache of cost-to-go Hessians
    p::Vector{V}   # Cache of cost-to-go gradients

    K::Vector{M}   # Cache of DDP feedback gains
    d::Vector{V}   # Cache of DDP feedforward terms

    function Problem(x₀, process::Process{T,V,D}, costs::Vector{C}, constraints::Vector{H}, M) where {T<:AbstractFloat,V<:AbstractVector{T},D<:Dynamics{T},C<:Cost{T},H<:Constraint{T}}
        Nk = process.Nk
        dynamics = process.dynamics
        @assert Nk == length(costs) "Number of costs doesn't match number of timesteps"
        @assert Nk == length(constraints) "Number of constraints doesn't match number of timesteps"

        z::Trajectory{T,V} = Trajectory(process)
        z̄::Trajectory{T,V} = Trajectory(process)
        v::ConstraintTrajectory{T,V} = ConstraintTrajectory(process, constraints)
        v̄::ConstraintTrajectory{T,V} = ConstraintTrajectory(process, constraints)

        A_d = Dict{Tuple{Int,Int},M}()
        B_d = Dict{Tuple{Int,Int},M}()

        cx_d = Dict{Tuple{Int,Int},M}()
        cu_d = Dict{Tuple{Int,Int},M}()

        lx_d = Dict{Int,V}()
        lu_d = Dict{Int,V}()
        lxx_d = Dict{Tuple{Int,Int},M}()
        luu_d = Dict{Tuple{Int,Int},M}()
        lux_d = Dict{Tuple{Int,Int},M}()

        Qx = Dict{Int,V}()
        Qu = Dict{Int,V}()
        Qxx = Dict{Tuple{Int,Int},M}()
        Quu = Dict{Tuple{Int,Int},M}()
        Qux = Dict{Tuple{Int,Int},M}()
        AtP = Dict{Tuple{Int,Int},M}()
        BtP = Dict{Tuple{Int,Int},M}()
        KtQu = Dict{Tuple{Int,Int},M}()
        λIμc = Dict{Int,V}()
        cxIμ = Dict{Tuple{Int,Int},M}()
        cuIμ = Dict{Tuple{Int,Int},M}()
        Quus = Dict{Tuple{Int,Int},M}()
        Iρ = Dict{Int,Diagonal{T,V}}()

        for k in 1:Nk
            n, m, c = nx(process.dynamics[k]), nu(process.dynamics[k]), nc(constraints[k])
            if !haskey(A_d, (n, n))
                A_d[(n, n)] = M(undef, n, n)
            end
            if !haskey(B_d, (n, m))
                B_d[(n, m)] = M(undef, n, m)
            end
            if !haskey(cx_d, (c, n))
                cx_d[(c, n)] = M(undef, c, n)
            end
            if !haskey(cu_d, (c, m))
                cu_d[(c, m)] = M(undef, c, m)
            end
            if !haskey(lx_d, n)
                lx_d[n] = V(undef, n)
            end
            if !haskey(lu_d, m)
                lu_d[m] = V(undef, m)
            end
            if !haskey(lxx_d, (n, n))
                lxx_d[(n, n)] = M(undef, n, n)
            end
            if !haskey(luu_d, (m, m))
                luu_d[(m, m)] = M(undef, m, m)
            end
            if !haskey(lux_d, (m, n))
                lux_d[(m, n)] = M(undef, m, n)
            end
            if !haskey(Qx, n)
                Qx[n] = V(undef, n)
            end
            if !haskey(Qu, m)
                Qu[m] = V(undef, m)
            end
            if !haskey(Qxx, (n, n))
                Qxx[(n, n)] = M(undef, n, n)
            end
            if !haskey(Quu, (m, m))
                Quu[(m, m)] = M(undef, m, m)
            end
            if !haskey(Qux, (m, n))
                Qux[(m, n)] = M(undef, m, n)
            end
            if !haskey(AtP, (n, n))
                AtP[(n, n)] = M(undef, n, n)
            end
            if !haskey(BtP, (m, n))
                BtP[(m, n)] = M(undef, m, n)
            end
            if !haskey(KtQu, (n, m))
                KtQu[(n, m)] = M(undef, n, m)
            end
            if !haskey(λIμc, c)
                λIμc[c] = V(undef, c)
            end
            if !haskey(cxIμ, (n, c))
                cxIμ[(n, c)] = M(undef, n, c)
            end
            if !haskey(cuIμ, (m, c))
                cuIμ[(m, c)] = M(undef, m, c)
            end
            if !haskey(Quus, (m, m))
                Quus[(m, m)] = M(undef, m, m)
            end
            if !haskey(Iρ, m)
                Iρ[m] = Diagonal(V(undef, m))
            end
        end

        A = [A_d[(nx(dynamics[k]), nx(dynamics[k]))] for k in 1:(Nk-1)]
        B = [B_d[(nx(dynamics[k]), nu(dynamics[k]))] for k in 1:(Nk-1)]

        cx = [cx_d[(nc(constraints[k]), nx(dynamics[k]))] for k in 1:Nk]
        cu = [cu_d[(nc(constraints[k]), nu(dynamics[k]))] for k in 1:Nk]

        lx = [lx_d[nx(dynamics[k])] for k in 1:Nk]
        lu = [lu_d[nu(dynamics[k])] for k in 1:Nk]
        lxx = [lxx_d[(nx(dynamics[k]), nx(dynamics[k]))] for k in 1:Nk]
        luu = [luu_d[(nu(dynamics[k]), nu(dynamics[k]))] for k in 1:Nk]
        lux = [lux_d[(nu(dynamics[k]), nx(dynamics[k]))] for k in 1:Nk]

        Qxd = [Qx[nx(process.dynamics[k])] for k in 1:Nk]
        Qud = [Qu[nu(process.dynamics[k])] for k in 1:Nk]
        Qxxd = [Qxx[(nx(process.dynamics[k]), nx(process.dynamics[k]))] for k in 1:Nk]
        Quud = [Quu[(nu(process.dynamics[k]), nu(process.dynamics[k]))] for k in 1:Nk]
        Quxd = [Qux[(nu(process.dynamics[k]), nx(process.dynamics[k]))] for k in 1:Nk]
        AtPd = [AtP[(nx(process.dynamics[k]), nx(process.dynamics[k]))] for k in 1:Nk]
        BtPd = [BtP[(nu(process.dynamics[k]), nx(process.dynamics[k]))] for k in 1:Nk]
        KtQud = [KtQu[(nx(process.dynamics[k]), nu(process.dynamics[k]))] for k in 1:Nk]
        λIμcd = [λIμc[nc(constraints[k])] for k in 1:Nk]
        cxIμd = [cxIμ[(nx(process.dynamics[k]), nc(constraints[k]))] for k in 1:Nk]
        cuIμd = [cuIμ[(nu(process.dynamics[k]), nc(constraints[k]))] for k in 1:Nk]
        Quusd = [Quus[(nu(process.dynamics[k]), nu(process.dynamics[k]))] for k in 1:Nk]
        Iρd = [Iρ[nu(process.dynamics[k])] for k in 1:Nk]

        P::Vector{M} = [M(undef, nx(dynamics[k]), nx(dynamics[k])) for k in 1:Nk]
        p::Vector{V} = [V(undef, nx(dynamics[k])) for k in 1:Nk]

        K::Vector{M} = [M(undef, nu(dynamics[k]), nx(dynamics[k])) for k in 1:Nk]
        d::Vector{V} = [V(undef, nu(dynamics[k])) for k in 1:Nk]

        new{T,V,M,D,C,H}(x₀,
            process, z, z̄, v, v̄,
            costs, constraints,
            A, B,
            cx, cu,
            lx, lu, lxx, luu, lux,
            Qxd, Qud, Qxxd, Quud, Quxd,
            AtPd, BtPd, KtQud, λIμcd,
            cxIμd, cuIμd,
            Quusd, Iρd,
            P, p,
            K, d)
    end

end

function rollout!(problem::Problem)
    rollout!(problem.process, problem.z, problem.x₀)
end

function rollout!(problem::Problem, U)
    rollout!(problem.process, problem.z, problem.x₀, U)
end

function eval_constraints!(problem::Problem, z::Trajectory, v::ConstraintTrajectory)
    constraints = problem.constraints
    c = v.c
    Nk = z.Nk
    X, U = z.X, z.U

    for k in 1:Nk
        constraint!(constraints[k], c[k], X[k], U[k])
    end
end

function eval_constraints!(problem::Problem)
    eval_constraints!(problem, problem.z, problem.v)
end

function constraint_violation(problem::Problem, v::ConstraintTrajectory)
    c = v.c
    constraints = problem.constraints
    Nk = problem.process.Nk
    cviol = 0.0

    for k in 1:Nk
        c_eq = @view c[k][1:nc_eq(constraints[k])]
        c_ineq = @view c[k][(nc_eq(constraints[k])+1):end]

        cviol_k_eq = maximum(c_eq, init=0.0)
        cviol_k_ineq = mapreduce(c -> clamp(c, 0.0, Inf), max, c_ineq)
        cviol = max(cviol, cviol_k_eq, cviol_k_ineq)
    end

    return cviol
end

function constraint_violation(problem::Problem)
    constraint_violation(problem, problem.v)
end

function eval_penalty_multiplier!(problem::Problem, v::ConstraintTrajectory, μ)
    constraints = problem.constraints
    c, λ, Iμ = v.c, v.λ, v.Iμ
    Nk = v.Nk

    for k in 1:Nk
        λ_ineq = @view λ[k][(nc_eq(constraints[k])+1):end]
        c_ineq = @view c[k][(nc_eq(constraints[k])+1):end]
        Iμ_eq = @view Iμ[k].diag[1:nc_eq(constraints[k])]
        Iμ_ineq = @view Iμ[k].diag[(nc_eq(constraints[k])+1):end]

        Iμ_eq .= μ
        map!((c, λ) -> ((c < 0.0) && (λ == 0.0)) ? 0.0 : μ, Iμ_ineq, c_ineq, λ_ineq)
    end
end

function eval_penalty_multiplier!(problem::Problem, μ)
    eval_penalty_multiplier!(problem, problem.p, μ)
end

function eval_cost(problem::Problem{T,V,M,D,C,H}, z::Trajectory{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T},D<:Dynamics{T},C<:Cost{T},H<:Constraint{T}}
    costs = problem.costs
    X, U = z.X, z.U
    Nk = z.Nk

    cost = 0.0
    for k in 1:Nk
        cost += value(costs[k], X[k], U[k])
    end

    return cost
end

function eval_cost(problem::Problem)
    return eval_cost(problem, problem.z)
end