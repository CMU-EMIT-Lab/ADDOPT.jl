
function build_U_wall(id::PlanarGMAWDynamics, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [(1 - 2mod(c + 1, 2)) * 0.0059; 0.0015 * (c - 1); 1.0; 0.0677]) # vx, vz, trim, WFS
        end

        push!(U, Uc)
    end

    return U
end

function build_U_wall(id::PlanarGMAWDynamicsPrescribed, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [0.0677]) # vx, vz, trim, WFS
        end

        push!(U, Uc)
    end

    return U
end

function build_U_wall(id::GMAWDynamicsPrescribed, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [0.0677]) # vx, vz, trim, WFS
        end

        push!(U, Uc)
    end

    return U
end

function build_U_wall(id::PlanarHeatsourceDynamics, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [0.002; 0.002; 250.0]) # vx, vz, P
        end

        push!(U, Uc)
    end

    return U
end

function build_U_wall(id::PlanarHeatsourcePrescribedMotionDynamics, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [80.0]) # P 250.0
        end

        push!(U, Uc)
    end

    return U
end

function rollout(process::Process, x₀, U, Nkb, Nkc, Nc, Δtb, Δtc; free_time=false)
    f!(dx, x, u, t, zi) = combined_dynamics!(dx, x, u, process, t, zi, Δtb)
    X = []
    x0 = copy(x₀)

    t = 0.0
    for c in 1:Nc
        zi = (c - 1) * (Nkb + Nkc)
        Xb = solve_RK4(f!, x0, U[c], Δtb, Nkb, t, zi)
        t += Δtb * Nkb
        Xc = solve_RK4(f!, step_RK4(f!, Xb[end], U[c][end], Δtb, t, zi + Nkb), [input_idle(process.input_dynamics) for k in 1:Nkc], Δtc, Nkc, t, zi + Nkb)
        t += Δtc * Nkc
        push!(X, vcat(Xb, Xc))
        x0 = step_RK4(f!, Xc[end], input_idle(process.input_dynamics), Δtc, t, zi + Nkb + Nkc)
    end

    return X
end

function rollout(process::Process, x₀, U, Nk, Δt)
    f!(dx, x, u, t, zi) = combined_dynamics!(dx, x, u, process, t, zi, Δt)
    X = solve_RK4(f!, x₀, U, Δt, Nk, 0.0, 0)

    return X
end

function marshall_z(idx, X, U, Δtb, Δtc; free_time=false)
    Nz, Nkb, Nkc, Nc, Nx, Nu = idx.Nz, idx.Nkb, idx.Nkc, idx.Nc, idx.Nx, idx.Nu

    z = zeros(Nz)

    for c in 1:Nc

        for k in 1:Nkb[c]
            z[idx.x[c][k]] .= X[c][k]
            z[idx.u[c][k]] .= U[c][k]

            if free_time
                z[idx.Δtb[c][k]] = Δtb
            end
        end

        for k in (Nkb[c]+1):(Nkb[c]+Nkc[c])
            z[idx.x[c][k]] .= X[c][k]

            if free_time
                z[idx.Δtc[c][k]] = Δtc
            end
        end
    end

    return z
end

function generate_wall_z₀(process::Process{ID,TD,PD}, idx, x₀, Δtb, Δtc; free_time=false) where {ID,TD,PD}
    Nz, Nkb, Nkc, Nc, Nx, Nu = idx.Nz, idx.Nkb, idx.Nkc, idx.Nc, idx.Nx, idx.Nu

    U = build_U_wall(process.input_dynamics, Nkb, Nc)
    X = rollout(process, x₀, U, Nkb, Nkc, Nc, Δtb, Δtc, free_time=free_time)

    z₀ = marshall_z(idx, X, U, Δtb, Δtc, free_time=free_time)

    return z₀
end

function initial_guess(process::Process{PlanarHeatsourceDynamics, PlanarVoxelTemperatureDynamics, NullPropertyDynamics}, mask, x₀, Nkc, Δtb, Δtc, T_melt, u)
    # generate initial guess
    x = copy(x₀)
    U0 = []
    X0 = []
    Nkb = 0

    f!(dx, x, u, t, zi) = combined_dynamics!(dx, x, u, process, t, zi, Δtb)

    while minimum(x[mask]) ≤ T_melt
        Nkb += 1

        push!(X0, x)
        push!(U0, u)
        x = step_RK4(f!, x, u, Δtb, 0.0, 1)
    end
    
    push!(X0, step_RK4(f!, X0[end], U0[end], Δtc, 0.0, 1))
    for k in 1:(Nkc-1)
        push!(X0, step_RK4(f!, X0[end], zeros(length(x₀)), Δtc, 0.0, 1))
    end

    U0 = [U0,]
    X0 = [X0,]

    return Nkb, U0, X0
end
