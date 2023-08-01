using ProgressMeter


# f!(dx, x, t)
# function solve_RK4(f!, x₀, dt, min_time, max_time)
#     K = trunc(Int, (max_time - min_time) / dt)
#     N̄ = length(x₀)
#     X = zeros(N̄, K)
#     X[:, 1] = x₀
#     k₁, k₂, k₃, k₄ = zeros(N̄), zeros(N̄), zeros(N̄), zeros(N̄)
#     x₁, x₂, x₃, x₄ = zeros(N̄), zeros(N̄), zeros(N̄), zeros(N̄)

#     try
#         @showprogress 0.5 "Simulating..." for (i, t) in enumerate(range(min_time, max_time, length=K)[1:(end-1)])
#             Xi = view(X, :, i)

#             x₁ .= Xi
#             f!(k₁, x₁, t)

#             @. x₂ = Xi + k₁ * dt / 2
#             f!(k₂, x₂, t + dt / 2)

#             @. x₃ = Xi + k₂ * dt / 2
#             f!(k₃, x₃, t + dt / 2)

#             @. x₄ = Xi + k₃ * dt
#             f!(k₄, x₄, t + dt)

#             @. X[:, i+1] = Xi + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt
#         end
#     catch e
#         println(showerror, e, catch_backtrace())
#     finally
#         return X
#     end

#     return X
# end


# f!(dx, x, u)
function solve_RK4(f!, x₀, U, dt, Nk, t₀, zi)
    Nx = length(x₀)
    X = [zeros(Nx) for k in 1:Nk]
    X[1] .= x₀
    k₁, k₂, k₃, k₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)
    x₁, x₂, x₃, x₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)

    # try
    t = t₀
    p = Progress(Nk - 1)
    # @showprogress 0.5 "Simulating..." 
    for i in 1:(Nk-1)
        Xi = X[i]
        Ui = U[i]

        x₁ .= Xi
        f!(k₁, x₁, Ui, t, zi + i)

        @. x₂ = Xi + k₁ * dt / 2
        f!(k₂, x₂, Ui, t, zi + i)

        @. x₃ = Xi + k₂ * dt / 2
        f!(k₃, x₃, Ui, t, zi + i)

        @. x₄ = Xi + k₃ * dt
        f!(k₄, x₄, Ui, t, zi + i)

        @. X[i+1] = Xi + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt
        next!(p)
        t += dt
    end
    # catch e
    #     println(showerror, e, catch_backtrace())
    # finally
    #     return X
    # end

    return X
end

function inplace_solve_rk4(process::Process{ID,TD,PD}, x₀::Vector{Ty}, dt::Float64, Nk::Int, t₀::Float64, zi::Int)::Vector{Ty} where {Ty,ID,TD,PD}
    Nx = length(x₀)
    x = zeros(Ty, Nx)
    u = zeros(Ty, Nx)
    x .= x₀
    k₁, k₂, k₃, k₄ = zeros(Ty, Nx), zeros(Ty, Nx), zeros(Ty, Nx), zeros(Ty, Nx)
    x₁, x₂, x₃, x₄ = zeros(Ty, Nx), zeros(Ty, Nx), zeros(Ty, Nx), zeros(Ty, Nx)

    t = t₀
    for i in 1:(Nk-1)
        x₁ .= x
        combined_dynamics!(k₁, x₁, u, process, t, zi + i)

        @. x₂ = x + k₁ * dt / 2
        combined_dynamics!(k₂, x₂, u, process, t, zi + i)

        @. x₃ = x + k₂ * dt / 2
        combined_dynamics!(k₃, x₃, u, process, t, zi + i)

        @. x₄ = x + k₃ * dt
        combined_dynamics!(k₄, x₄, u, process, t, zi + i)

        x .+= @. (k₁ + 2k₂ + 2k₃ + k₄) * (dt / 6)
        t += dt
    end

    # @show x[1:(Nx÷3)]
    # @show x[(end-Nx÷3+1):end]
    return x
end

function step_RK4(f!, Xi, u, dt, t, zi)
    Nx = length(Xi)
    k₁, k₂, k₃, k₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)
    x₁, x₂, x₃, x₄ = zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx)
    Xf = zeros(Nx)

    x₁ .= Xi
    f!(k₁, x₁, u, t, zi)

    @. x₂ = Xi + k₁ * dt / 2
    f!(k₂, x₂, u, t + dt / 2, zi)

    @. x₃ = Xi + k₂ * dt / 2
    f!(k₃, x₃, u, t + dt / 2, zi)

    @. x₄ = Xi + k₃ * dt
    f!(k₄, x₄, u, t + dt, zi)

    @. Xf = Xi + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * dt

    return Xf
end

function reinterpolate_traj(t)
end