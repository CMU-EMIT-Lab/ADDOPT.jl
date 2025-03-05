using ADDOPT
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using CSV, Tables
using Printf
using JLD2

HVmin = 135.0
HVmax = 404.0
# HVmin = 140.0
# HVmax = 400.0
y_init = (HVmax - 365) / (HVmax - HVmin)
# y_init = (HVmax - 380) / (HVmax - HVmin)

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

# Geometric parameters
mask_img = Ty.(Gray.(load("examples/ebpbf/scotty_bw.png")))

n_subsample = 5
mask_img_blurred = imfilter(mask_img, Kernel.gaussian(n_subsample))
mask_img_subsampled = mask_img_blurred[(n_subsample÷2):n_subsample:end, (n_subsample÷2):n_subsample:end]
mask_img_subsampled = (1.0 .- Float64.(mask_img_subsampled)) .* (0.9 - y_init) .+ y_init

mask_top = vec(mask_img_subsampled)
Nx, Ny = size(mask_img_subsampled)
Nz = 4
buffer = 2
nsvox = Ny * Nx
nvox = nsvox * Nz
Nu = (Nx - 2buffer) * (Ny - 2buffer)
mask = vcat(mask_top, y_init * ones(Ty, nsvox * (Nz - 1)))

l = 1e-3 * n_subsample / 7 # m

@show Nx, Ny
@show nsvox
@show nvox

η = 1.0
σ = 1200e-6 / 1.35 # Spot diameter, m
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0 * η # W

# Material parameters, taken at the solidus
k = 31.1e3 # W / kK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0e3 # J / kg kK 

# Tempering parameters
lnA::Ty = 8.845 # 38.005
n::Ty = 0.358 # 0.051590
E::Ty = 56.277e-3 # 240.24 # kJ / mol K

# lnA::Ty = 38.005
# n::Ty = 0.051590
# E::Ty = 240.24e-3 # kJ / mol K

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 
T_AC1 = 1000.0e-3 # kK

# Environment parameters
T∞ = (20.0 + Pₛₑₜ * 0.100 * 50 / ((1 / 2) * (0.1)^2 * sqrt(k / ρ / cₚ * 5.0) * ρ * cₚ) + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)

dt::Ty = 50e-3 # s, aka 50ms

Tmax = T_AC1 * ones(nvox)

tempering = Tempering(lnA, n, E, dt; num=nvox)
pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; σ=σ, buffer=buffer)
dynamics = PBFTempering(pbf_powerfield, tempering)

power_mask = zeros(Bool, (Nx, Ny, Nz))
power_mask[(1+buffer):(end-buffer), (1+buffer):(end-buffer), 1] .= true
power_mask = vec(power_mask)

x₀ = V([T∞ * ones(nvox); log(-log(1 - y_init)) * ones(nvox)])
uw = (mask_top[power_mask[1:nsvox]] .+ 0.2) * Pₛₑₜ / sum((mask_top[power_mask[1:nsvox]] .+ 0.2))
uw = V(uw)
u0 = zeros(Nu)
u0 = V(u0)

Nk = 0
Nkc = 0
x1, x2 = copy(x₀), copy(x₀)
for i in 1:1000
    transition!(dynamics, x2, x1, uw)
    if maximum(x2) ≥ T_AC1 - 100e-3
        break
    end
    Nk = i
    x1 .= x2
end
for i in 1:1000
    transition!(dynamics, x2, x1, u0)
    if maximum(x2) ≤ 450e-3
        break
    end
    Nkc = i
    x1 .= x2
end
@show Nk = Nk + Nkc

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, 2nvox)
b_x_eq = V(undef, 0)
A_u_eq = M(ones(1, Nu) ./ Pₛₑₜ)
b_u_eq = V([1.0])
A_x_ineq = M([collect(1.0 * I(nvox)) zeros(nvox, nvox)])
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(Nu)))
b_u_ineq = V(zeros(Nu))

build_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
cooling_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_ineq, b_u_ineq, A_x_ineq, b_x_ineq, M(undef, 0, Nu), V(undef, 0))
constraints = vcat([build_constraint for _ in 1:(Nk-Nkc)], [cooling_constraint for _ in 1:Nkc])


x̄ = V([T∞ * ones(nvox); log.(-log.(1 .- mask))])
ū = V(zeros(Nu))

Q = diagm([zeros(nvox); 1e-2 / Nk / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
Qf = diagm([zeros(nvox); 1e3 / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
R = diagm(0.0 * ones(Nu))
Q = M(Q)
Qf = M(Qf)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)
costs = [cost for _ in 1:(Nk-1)]
final_cost = QuadraticCost(Qf, R, x̄, ū)
push!(costs, final_cost)

problem = Problem(x₀, process, costs, constraints, M)
U0 = vcat([uw for k in 1:(Nk-Nkc)], [u0 for k in 1:Nkc])


for iter in 1:1#500
    # Generate random initial guess
    # for k in 1:(Nk-Nkc)
    #     CUDA.rand!(U0[k])
    #     U0[k] .*= Pₛₑₜ / sum(U0[k])
    # end

    @time rollout!(problem, U0)
    @time rollout!(problem, U0)
    eval_constraints!(problem, problem.z, problem.v)
    eval_penalty_multiplier!(problem, problem.v, 1e-1)
    @show constraint_violation(problem, problem.v)
    eval_lagrangian_cost(problem.v)
    U_naive = [Array(u) for u in problem.z.U]
    save_object("initialguess_U_$(iter)_$(n_subsample).jld2", U_naive)

    ### VISUALIZE BEFORE OPTIMIZATION ###
    X, U = problem.z.X, problem.z.U
    t = (0:(Nk-1)) .* dt
    T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X]
    ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
    P_surface = [reverse(reshape(Array((pbf_powerfield.B*u)[1:nsvox] .* (l^3 * ρ * cₚ) ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U]
    Pdmax = round(maximum([maximum(P) for P in P_surface]), sigdigits=1)

    y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]
    heatmap(abs.(y_surface[end] .- reverse(mask_img_subsampled, dims=1)), aspect_ratio=:equal, title="Error Magnitude")

    anim = @animate for (t, P, T, y) in zip(t, P_surface, T_surface, y_surface)
        title_str = @sprintf "Process at %.3f seconds" t

        h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, Pdmax))
        h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_AC1 * 1e3), title=title_str)
        h3 = heatmap(y, aspect_ratio=:equal, ticks=false, c=:acton, cbar_title="Fraction Tempered", clim=(0, 1))

        plot(h1, h2, h3; layout=(1, 3), size=((12Nx) * 3, 12Ny + 110))
    end
    gif(anim, "unoptimized_$(iter)_$(n_subsample).mp4", fps=(1 / dt))

    ### OPTIMIZE ###

    @time rollout!(problem, U0)
    @show eval_cost(problem)
    @time al_ddp!(problem; ctol=1e-4, μ=0.1, ϕ=3.0, verbosity=2, ρi=1e-10, tol=1e-4, gtol=Pₛₑₜ / Nu / 100)
    @time al_ddp!(problem; ctol=1e-4, μ=0.1, ϕ=3.0, verbosity=2, ρi=1e-10, tol=1e-4, gtol=Pₛₑₜ / Nu / 100)
    @show eval_cost(problem)
    U_opt = [Array(u) for u in problem.z.U]
    save_object("solution_U_$(iter)_$(n_subsample).jld2", U_opt)

    ### VISUALIZE AFTER OPTIMIZATION ###
    X, U = problem.z.X, problem.z.U
    t = (0:(Nk-1)) .* dt
    T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X]
    ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
    P_surface = [reverse(reshape(Array((pbf_powerfield.B*u)[1:nsvox] .* (l^3 * ρ * cₚ) ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U]
    Pdmax = round(maximum([maximum(P) for P in P_surface]), sigdigits=1)

    y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]

    anim = @animate for (t, P, T, y) in zip(t, P_surface, T_surface, y_surface)
        title_str = @sprintf "Process at %.3f seconds" t

        h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, Pdmax))
        h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_AC1 * 1e3), title=title_str)
        h3 = heatmap(y, aspect_ratio=:equal, ticks=false, c=:acton, cbar_title="Fraction Tempered", clim=(0, 1))

        plot(h1, h2, h3; layout=(1, 3), size=((12Nx) * 3, 12Ny + 110))
    end
    gif(anim, "optimized_$(iter)_$(n_subsample).mp4", fps=(1 / dt))
end

# ### APPROXIMATE WITH SPOTS ###
# seq, dwell = powerfield_to_sequence(dt, 1 / ω, 1e-6, [Array(u) for u in U[1:(end-Nkc)]])

# process = problem = 0
# tempering_sim = Tempering(lnA, n, E, Ty(dwell); num=nvox)
# pbf_powerfield_sim = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, Ty(dwell), Ty, V, M; σ=σ, buffer=buffer)
# dynamics_sim = PBFTempering(pbf_powerfield_sim, tempering_sim)

# U_spot = vcat([zeros(Nu) for _ in 1:3length(seq)])
# for (u, idx) in zip(U_spot, seq)
#     u[idx] = Pₛₑₜ
# end
# U_spot = [V(u) for u in U_spot]
# X_spot = [V(zeros(2nvox)) for u in U_spot]

# rollout!([dynamics_sim for _ in U_spot], 3 * length(seq), x₀, X_spot, U_spot)

# ### VISUALIZE AFTER OPTIMIZATION ###
# t = (0:(length(U_spot)-1)) .* dwell
# T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X_spot]
# ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X_spot]
# P_surface = [reverse(reshape(Array((pbf_powerfield_sim.B*u)[1:nsvox] .* (l^3 * ρ * cₚ) ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U_spot]

# y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]

# heatmap(abs.(y_surface[end] .- reverse(mask_img_subsampled, dims=1)), aspect_ratio=:equal, title="Error Magnitude")

# skip = 100
# for k in (skip+1):skip:(length(U_spot))
#     P_surface[k] .= sum(P_surface[(k-skip):(k-1)]) ./ skip
# end
# anim = @animate for (t, P, T, y) in zip(t[1:skip:end], P_surface[1:skip:end], T_surface[1:skip:end], y_surface[1:skip:end])
#     title_str = @sprintf "Process at %.3f seconds" t

#     h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, Pdmax))
#     h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_AC1 * 1e3), title=title_str)
#     h3 = heatmap(y, aspect_ratio=:equal, ticks=false, c=:acton, cbar_title="Fraction Tempered", clim=(0, 1))

#     plot(h1, h2, h3; layout=(1, 3), size=((12Nx) * 3, 12Ny + 110))
# end
# gif(anim, "approximated_$(n_subsample).mp4", fps=(1 / dwell / skip))

# xyz = get_coordinates(Nx - 2buffer, Ny - 2buffer, 1, l)
# points = [(xyz[:, seq][1], xyz[:, seq][2]) for seq in seq]
# points
# CSV.write("scan_strat_optimized.csv", Tables.table(vcat([[round(x, digits=9); round(y, digits=9); round(dwell, digits=9)]' for (x, y) in points]...); header=["X", "Y", "Δt"]))