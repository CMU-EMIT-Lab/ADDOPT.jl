
function build_U_wall(id::PlanarGMAWDynamics, Nk, Nc)
    U = []

    for c in 1:Nc
        Uc = []
        for k in 1:Nk
            push!(Uc, [(1 - 2mod(c, 2)) * 0.0059; 0.0; 1.0; 0.0677]) # vx, vz, trim, WFS
        end

        push!(U, Uc)
    end

    return U
end

function rollout(process::Process, x₀, U, Nk, Nc, Δt, tc; free_time=false)
    f!(dx, x, u) = combined_dynamics!(dx, x, u, process)
    X = []

    for c in 1:Nc
        Xc = solve_RK4(f!, x₀, U[c], Δt, Nk)
        push!(X, Xc)
        x₀ = copy(x₀)
        combined_jump!(x₀, Xc[Nk], tc[c], process)
    end

    return X
end

function marshall_z(idx, X, U, Δt, tc; free_time=false)
    Nz, Nk, Nc, Nx, Nu = idx.Nz, idx.Nkb, idx.Nc, idx.Nstates, idx.Nu

    z = zeros(Nz)

    for c in 1:Nc
        z[idx.tc[c]] = tc[c]

        for k in 1:Nk
            z[idx.x[c][k]] = X[c][k]
            z[idx.u[c][k]] = U[c][k]

            if free_time
                z[idx.Δt[c][k]] = Δt
            end
        end
    end

    return z
end

function generate_wall_z₀(process::Process{PlanarGMAWDynamics, PlanarVoxelMassEnergyDynamics, PD}, idx, x₀, Δt, tc; free_time=false) where {PD <: PropertyDynamics}
    Nz, Nk, Nc, Nx, Nu = idx.Nz, idx.Nk, idx.Nc, idx.Nstates, idx.Nu

    U = build_U_wall(process.id, Nk, Nc)
    X = rollout(process, x₀, U, Nk, Nc, Δt, tc, free_time=free_time)

    z₀ = marshall_z(idx, X, U, Δt, tc, free_time=free_time)

    return z₀
end

