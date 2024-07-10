
function al_ilqr!(problem::Problem; μ=0.1, ϕ=10.0, maxiters=100, inneriters=400, tol=1e-4, ctol=1e-6, verbosity=1, ρi=1e-10)
    process = problem.process
    N = process.Nk
    λ, c = problem.v.λ, problem.v.c
    constraints = problem.constraints

    # Clear Lagrange multipliers
    for k in 1:N
        λ[k] .= 0
    end

    for iter in 1:maxiters
        J = ilqr!(problem, μ; ilqr_iters=inneriters, tol=tol, verbosity=verbosity, ρi=ρi)
        cviol = constraint_violation(problem)
        if verbosity ≥ 1
            @printf "AL-iLQR iteration %03d - J: %.5e - |c|∞: %.5e - μ: %.4e\n" iter J cviol μ
        end
        # @info "$iter, ΔJ: $(round(ΔJ, digits=4)), μ: $(round(μ, digits=4))"

        if cviol < ctol
            return
        end

        # Update Lagrange multipliers
        for k in 1:N
            λ[k] .+= μ .* c[k]

            λ_ineq = @view λ[k][(nc_eq(constraints[k])+1):end]
            clamp!(λ_ineq, 0.0, Inf)
        end

        # Increase penalty
        μ *= ϕ
        μ = clamp(μ, 1.0, 1e8)
    end

    @warn "Exceeded maximum iterations for AL-iLQR"
end

function ilqr!(problem::Problem, μ; ilqr_iters=300, tol=1e-4, verbosity=2, ρi=1e-10)
    rollout!(problem)
    eval_constraints!(problem)
    eval_penalty_multiplier!(problem, problem.v, μ)

    for iter in 1:ilqr_iters
        J_old = eval_augmented_cost(problem, problem.z, problem.v)
        ilqr_step!(problem, μ; verbosity=verbosity, ρi=ρi)
        J_new = eval_augmented_cost(problem, problem.z, problem.v)
        if verbosity ≥ 2
            @printf "iLQR iteration %03d - J: %.5e\n" iter J_new
        end

        if abs(J_old - J_new) < tol
            return J_new
        end
    end

    @warn "Exceed maximum iterations for iLQR"
    return -1.0
end

function ilqr_step!(problem::Problem, μ; ρi=1e-8, verbosity=3)
    ρ = 0.0
    dρ = 0.0
    ΔJ = Inf

    eval_penalty_multiplier!(problem, problem.v, μ)

    for _ in 1:100
        ## Backward pass: calculate gains and feedforward term
        ΔV₁, ΔV₂ = backward_pass(problem, ρ; verbosity=verbosity)
        while isinf(ΔV₁)
            dρ = max(dρ * 1.6, 1.6)
            ρ = max(ρ * dρ, ρi)
            ρ = clamp(ρ, 0.0, 1e8)
            if verbosity ≥ 3
                println("Increasing Regularization after Backward Pass Failure, ρ: $ρ")
            end
            ΔV₁, ΔV₂ = backward_pass(problem, ρ; verbosity=verbosity)
        end

        ## Rollout with line search
        ΔJ = forward_pass(problem, ΔV₁, ΔV₂, μ; verbosity=verbosity)

        if ΔJ != Inf
            break
        end

        ## Increase Regularization
        dρ = max(dρ * 1.6, 1.6)
        ρ = max(ρ * dρ, ρi)
        ρ = clamp(ρ, 0.0, 1e8)
        if verbosity ≥ 3
            println("Increasing Regularization after Line Search Failure, ρ: $ρ")
        end
    end

    return ΔJ
end

function backward_pass(problem::Problem, ρ; verbosity=4)
    # Get necessary variables
    dynamics = problem.process.dynamics
    constraints = problem.constraints
    costs = problem.costs
    x, u = problem.z.X, problem.z.U
    v = problem.v
    c, λ, Iμ = v.c, v.λ, v.Iμ
    A, B = problem.A, problem.B
    cx, cu = problem.cx, problem.cu
    lx, lu, lxx, luu, lux = problem.lx, problem.lu, problem.lxx, problem.luu, problem.lux
    Qxd, Qud, Qxxd, Quud, Quxd, = problem.Qx, problem.Qu, problem.Qxx, problem.Quu, problem.Qux
    AtPd, BtPd, KtQud, λIμcd, = problem.AtP, problem.BtP, problem.KtQu, problem.λIμc
    cxIμd, cuIμd = problem.cxIμ, problem.cuIμ
    Quu_scratchd, Iρd = problem.Quus, problem.Iρ
    P, p = problem.P, problem.p
    K, d = problem.K, problem.d
    N = v.Nk

    ΔV₁ = 0.0
    ΔV₂ = 0.0

    # Optimal terminal cost-to-go second order expansion
    # p[N] .= lx[N] .+ cx[N]' * (λ[N] + Iμ[N] * c[N])
    λIμc = λIμcd[N]
    λIμc .= λ[N]
    mul!(λIμc, Iμ[N], c[N], 1.0, 1.0)
    p[N] .= lx[N]
    mul!(p[N], cx[N]', λIμc, 1.0, 1.0)
    # P[N] .= lxx[N] .+ cx[N]' * Iμ[N] * cx[N]
    cxIμ = cxIμd[N]
    mul!(cxIμ, cx[N]', Iμ[N])
    P[N] .= lxx[N]
    mul!(P[N], cxIμ, cx[N], 1.0, 1.0)

    for k in (N-1):-1:1
        # Update derivatives
        transition_state_jacobian!(dynamics[k], A[k], x[k], u[k])
        transition_input_jacobian!(dynamics[k], B[k], x[k], u[k])
        constraint_state_jacobian!(constraints[k], cx[k], x[k], u[k])
        constraint_input_jacobian!(constraints[k], cu[k], x[k], u[k])
        cost_state_gradient!(costs[k], lx[k], x[k], u[k])
        cost_input_gradient!(costs[k], lu[k], x[k], u[k])
        cost_state_hessian!(costs[k], lxx[k], x[k], u[k])
        cost_input_hessian!(costs[k], luu[k], x[k], u[k])
        cost_input_state_hessian!(costs[k], lux[k], x[k], u[k])

        # Pull working matrices / vectors from cache
        Qx, Qu, Qxx, Quu, Qux = Qxd[k], Qud[k], Qxxd[k], Quud[k], Quxd[k]
        AtP, BtP, KtQu, λIμc = AtPd[k], BtPd[k], KtQud[k], λIμcd[k]
        cxIμ, cuIμ = cxIμd[k], cuIμd[k]
        Quu_scratch, Iρ = Quu_scratchd[k], Iρd[k]

        mul!(AtP, A[k]', P[k+1])
        mul!(BtP, B[k]', P[k+1])
        # λ[k] + Iμ[k] * c[k]
        λIμc .= λ[k]
        mul!(λIμc, Iμ[k], c[k], 1.0, 1.0)

        mul!(cxIμ, cx[k]', Iμ[k])
        mul!(cuIμ, cu[k]', Iμ[k])

        # Qxx .= lxx[k] .+ A[k]' * P[k+1] * A[k] .+ cx[k]' * Iμ[k] * cx[k]
        Qxx .= lxx[k]
        mul!(Qxx, AtP, A[k], 1.0, 1.0)
        mul!(Qxx, cxIμ, cx[k], 1.0, 1.0)

        # Quu .= luu[k] .+ B[k]' * P[k+1] * B[k] .+ cu[k]' * Iμ[k] * cu[k]
        Quu .= luu[k]
        mul!(Quu, BtP, B[k], 1.0, 1.0)
        mul!(Quu, cuIμ, cu[k], 1.0, 1.0)

        # Qux .= lux[k] .+ B[k]' * P[k+1] * A[k] .+ cu[k]' * Iμ[k] * cx[k]
        Qux .= lux[k]
        mul!(Qux, BtP, A[k], 1.0, 1.0)
        mul!(Qux, cuIμ, cx[k], 1.0, 1.0)

        # Qx = lx[k] + A[k]' * p[k+1] + cx[k]' * (λ[k] + Iμ[k] * c[k])
        Qx .= lx[k]
        mul!(Qx, A[k]', p[k+1], 1.0, 1.0)
        mul!(Qx, cx[k]', λIμc, 1.0, 1.0)

        # Qu = lu[k] + B[k]' * p[k+1] + cu[k]' * (λ[k] + Iμ[k] * c[k])
        Qu .= lu[k]
        mul!(Qu, B[k]', p[k+1], 1.0, 1.0)
        mul!(Qu, cu[k]', λIμc, 1.0, 1.0)

        # Regularization
        Iρ.diag .= ρ
        Quu[1:(size(Quu, 1)+1):end] .+= Iρ.diag
        Quu_scratch .= Quu
        Quu_scratch .+= Quu'
        Quu_scratch ./= 2.0
        # Quu_scratch = Symmetric(Quu)
        # Quu_scratch .+= Iρ
        Quu_factorized = LinearAlgebra.cholesky!(Quu_scratch, check=false)

        # C, info = LinearAlgebra._chol!(Quu_scratch, UpperTriangular)
        # Quu_factorized = Cholesky(C.data, 'L', info)
        if !issuccess(Quu_factorized) # !isposdef(Quu_factorized)#
            if verbosity ≥ 4
                println("Failing backward pass factorization at iteration $k")
            end
            return -Inf, -Inf
        end
        # Quu_factorized = Quu_scratch
        # K[k] .= Quu \ Qux
        ldiv!(K[k], Quu_factorized, Qux)
        K[k] .*= -1
        # d[k] .= Quu \ Qu
        ldiv!(d[k], Quu_factorized, Qu)
        d[k] .*= -1

        mul!(KtQu, K[k]', Quu)

        # P[k] .= Qxx .+ K[k]' * Quu * K[k] .+ K[k]' * Qux .+ Qux' * K[k]
        P[k] .= Qxx
        mul!(P[k], KtQu, K[k], 1.0, 1.0)
        mul!(P[k], K[k]', Qux, 1.0, 1.0)
        mul!(P[k], Qux', K[k], 1.0, 1.0)

        # p[k] .= Qx .+ K[k]' * Quu * d[k] .+ K[k]' * Qu .+ Qux' * d[k]
        p[k] .= Qx
        mul!(p[k], KtQu, d[k], 1.0, 1.0)
        mul!(p[k], K[k]', Qu, 1.0, 1.0)
        mul!(p[k], Qux', d[k], 1.0, 1.0)

        ΔV₁ += dot(d[k], Qu)
        ΔV₂ += 0.5 * dot(d[k], Quu * d[k])#dot(d[k], Quu, d[k])
    end

    return ΔV₁, ΔV₂
end

function forward_pass(problem, ΔV₁, ΔV₂, μ; verbosity=4)
    return linesearch_on_rollout!(problem, ΔV₁, ΔV₂, μ; verbosity=verbosity)
end

function linesearch_on_rollout!(problem::Problem, ΔV₁, ΔV₂, μ; α=1.0, c=0.5, maxiters=20, verbosity=4)
    z, z̄ = problem.z, problem.z̄
    v, v̄ = problem.v, problem.v̄

    J0 = eval_augmented_cost(problem, z, v)
    for k in 1:v.Nk
        v̄.λ[k] .= v.λ[k]
    end

    if verbosity ≥ 4
        println("Line Search -- J0: $J0")
        println("i\tlog(α)\t\tJ0\t\tJ\tΔJ̄")
    end

    for i in 1:maxiters
        ddp_rollout!(problem, α, z, z̄; verbosity=verbosity)

        eval_constraints!(problem, z̄, v̄)
        eval_penalty_multiplier!(problem, v̄, μ)

        J = eval_augmented_cost(problem, z̄, v̄)

        ΔJ̄ = -α * ΔV₁ - (α^2) * ΔV₂

        if verbosity ≥ 4
            @printf "%d\t%.2f\t%.4e\t%.4e\t%.4e\n" i log(α) J0 J ΔJ̄
        end
        # if 0.0 < ΔJ̄ < 1e-8
        #     return Inf
        # end

        if ΔJ̄ > 0.0 && 1e-4 ≤ (J0 - J) / ΔJ̄
            copy!(z, z̄)
            copy!(v, v̄)
            # println("Linesearch suceeded with step of $(J0 - J)")
            return J0 - J
        end

        α *= c
    end

    return Inf
end

function ddp_rollout!(problem::Problem{T,V,M,D}, α, z::Trajectory{T,V}, z̄::Trajectory{T,V}; verbosity=5) where {T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T},D<:Dynamics{T}}
    process = problem.process
    dynamics = process.dynamics
    N = process.Nk
    X, U = z.X, z.U
    X̄, Ū = z̄.X, z̄.U
    K, d = problem.K, problem.d

    X̄[1] .= problem.x₀

    for k in 1:(N-1)
        ddp_rollout_step!(dynamics[k], X̄[k+1], X[k], X̄[k], U[k], Ū[k], α, d[k], K[k]; verbosity=verbosity)
    end
end

function ddp_rollout_step!(dynamics::Dynamics, x̄ₖ₊₁, xₖ, x̄ₖ, uₖ, ūₖ, α, d, K; verbosity=5) #where {T,D<:Dynamics{T}}
    # Ū[k] = U[k] + α * d[k] + K[k] * (X̄[k] - X[k])
    ūₖ .= uₖ .+ α .* d
    mul!(ūₖ, K, x̄ₖ, 1.0, 1.0)
    mul!(ūₖ, K, xₖ, -1.0, 1.0)
    if verbosity ≥ 5
        @show sum(ūₖ)
    end
    transition!(dynamics, x̄ₖ₊₁, x̄ₖ, ūₖ)
end

function eval_augmented_cost(problem, z::Trajectory, v::ConstraintTrajectory)
    return eval_cost(problem, z) + eval_lagrangian_cost(v)
end
