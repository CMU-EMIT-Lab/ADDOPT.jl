using ProgressMeter


# f!(dx, x, t)
function solve_RK4(f!, x₀, dt, min_time, max_time)
    K = trunc(Int, (max_time - min_time) / dt)
    N̄ = length(x₀)
    X = zeros(N̄, K)
    X[:, 1] = x₀
    k₁, k₂, k₃, k₄ = zeros(N̄), zeros(N̄), zeros(N̄), zeros(N̄)
    x₁, x₂, x₃, x₄ = zeros(N̄), zeros(N̄), zeros(N̄), zeros(N̄)

    try
        @showprogress 0.5 "Simulating..." for (i, t) in enumerate(range(min_time, max_time, length=K)[1:(end-1)])
            Xi = view(X, :, i)

            x₁ .= Xi
            f!(k₁, x₁, t)

            @. x₂ = Xi + k₁ * dt / 2
            f!(k₂, x₂, t + dt / 2)

            @. x₃ = Xi + k₂ * dt / 2
            f!(k₃, x₃, t + dt / 2)

            @. x₄ = Xi + k₃ * dt
            f!(k₄, x₄, t + dt)

            @. X[:, i+1] = Xi + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt
        end
    catch e
        println(showerror, e, catch_backtrace())
    finally
        return X
    end

    return X
end


# f!(dx, x, u)
function solve_RK4(f!, x₀, U, dt, Nk)
    Nx = length(x₀)
    X = [zeros(Nx) for k in 1:Nk]
    X[1] .= x₀
    k₁, k₂, k₃, k₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)
    x₁, x₂, x₃, x₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)

    try
        @showprogress 0.5 "Simulating..." for i in 1:Nk
            Xi = view(X, i)
            Ui = view(U, i)

            x₁ .= Xi
            f!(k₁, x₁, Ui)

            @. x₂ = Xi + k₁ * dt / 2
            f!(k₂, x₂, Ui)

            @. x₃ = Xi + k₂ * dt / 2
            f!(k₃, x₃, Ui)

            @. x₄ = Xi + k₃ * dt
            f!(k₄, x₄, Ui)

            @. X[i+1] = Xi + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt
        end
    catch e
        println(showerror, e, catch_backtrace())
    finally
        return X
    end

    return X
end